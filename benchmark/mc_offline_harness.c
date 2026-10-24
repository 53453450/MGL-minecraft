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
 * Measured: per-frame wall/CPU, getrusage user/sys split, and — when linked
 * against an MGL build carrying the mgl_vm_probe_count counter — exact
 * vm_region probe counts per frame (dlsym, optional).
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

#define VAO_CACHE_SIZE 12           /* VertexArrayCache keyed by format; few formats in practice */
#define UBO_RING_BUFFERS 2          /* MappableRingBuffer(…, 130, …) */
#define UBO_BLOCK_ALIGN 256         /* GlDevice uniform offset alignment */
#define TRANSFORM_UBO_SIZE 96       /* DynamicUniforms.TRANSFORM_UBO_SIZE (std140) */

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
    double cpu_per_frame_ms;
    double user_per_frame_ms;
    double sys_per_frame_ms;
    double p50_ms;
    double p95_ms;
} FrameStats;

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static void summarize(FrameStats *out, const double *wall_ms, const double *user_ms,
                      const double *sys_ms, int n)
{
    double *sorted = malloc((size_t)n * sizeof(double));
    double wall = 0.0, user = 0.0, sys = 0.0;
    for (int i = 0; i < n; i++) {
        wall += wall_ms[i];
        user += user_ms[i];
        sys += sys_ms[i];
        sorted[i] = wall_ms[i];
    }
    qsort(sorted, (size_t)n, sizeof(double), cmp_double);
    out->wall_per_frame_ms = wall / n;
    out->user_per_frame_ms = user / n;
    out->sys_per_frame_ms = sys / n;
    out->cpu_per_frame_ms = out->user_per_frame_ms + out->sys_per_frame_ms;
    out->p50_ms = sorted[(int)(n * 0.50)];
    out->p95_ms = sorted[(int)(n * 0.95)];
    free(sorted);
}

/* ------------------------------------------------------------------ */
/* GL loader (same pattern as mgl_benchmark.c)                          */
/* ------------------------------------------------------------------ */

/* glcorearb.h stops short of the GL 4.3 glVertexAttribIFormat typedef. */
typedef void (APIENTRYP MGL_PFNGLVERTEXATTRIBIFORMATPROC)(GLuint index, GLint size,
                                                          GLenum type, GLuint relativeoffset);
static MGL_PFNGLVERTEXATTRIBIFORMATPROC p_glVertexAttribIFormat;

static PFNGLBINDBUFFERPROC p_glBindBuffer;
static PFNGLBINDBUFFERRANGEPROC p_glBindBufferRange;
static PFNGLBUFFERSUBDATAPROC p_glBufferSubData;
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
static PFNGLVIEWPORTPROC p_glViewport;
static PFNGLUSEPROGRAMPROC p_glUseProgram;
static PFNGLGETUNIFORMLOCATIONPROC p_glGetUniformLocation;
static PFNGLUNIFORM1IPROC p_glUniform1i;
static PFNGLGETBUFFERPARAMETERIVPROC p_glGetBufferParameteriv;

#define LOAD_GL(ptr, type, name) \
    do { \
        ptr = (type)glfwGetProcAddress(name); \
        if (!ptr) { fprintf(stderr, "missing GL entry %s\n", name); return 1; } \
    } while (0)

static int load_gl(void)
{
    LOAD_GL(p_glBindBuffer, PFNGLBINDBUFFERPROC, "glBindBuffer");
    LOAD_GL(p_glBindBufferRange, PFNGLBINDBUFFERRANGEPROC, "glBindBufferRange");
    LOAD_GL(p_glBufferSubData, PFNGLBUFFERSUBDATAPROC, "glBufferSubData");
    LOAD_GL(p_glBufferData, PFNGLBUFFERDATAPROC, "glBufferData");
    LOAD_GL(p_glDrawArrays, PFNGLDRAWARRAYSPROC, "glDrawArrays");
    LOAD_GL(p_glDrawElements, PFNGLDRAWELEMENTSPROC, "glDrawElements");
    LOAD_GL(p_glBindVertexArray, PFNGLBINDVERTEXARRAYPROC, "glBindVertexArray");
    LOAD_GL(p_glBindVertexBuffer, PFNGLBINDVERTEXBUFFERPROC, "glBindVertexBuffer");
    LOAD_GL(p_glVertexAttribFormat, PFNGLVERTEXATTRIBFORMATPROC, "glVertexAttribFormat");
    LOAD_GL(p_glVertexAttribBinding, PFNGLVERTEXATTRIBBINDINGPROC, "glVertexAttribBinding");
    LOAD_GL(p_glVertexAttribIFormat, MGL_PFNGLVERTEXATTRIBIFORMATPROC, "glVertexAttribIFormat");
    LOAD_GL(p_glEnableVertexAttribArray, PFNGLENABLEVERTEXATTRIBARRAYPROC, "glEnableVertexAttribArray");
    LOAD_GL(p_glGenVertexArrays, PFNGLGENVERTEXARRAYSPROC, "glGenVertexArrays");
    LOAD_GL(p_glBindTexture, PFNGLBINDTEXTUREPROC, "glBindTexture");
    LOAD_GL(p_glActiveTexture, PFNGLACTIVETEXTUREPROC, "glActiveTexture");
    LOAD_GL(p_glBindSampler, PFNGLBINDSAMPLERPROC, "glBindSampler");
    LOAD_GL(p_glBindFramebuffer, PFNGLBINDFRAMEBUFFERPROC, "glBindFramebuffer");
    LOAD_GL(p_glViewport, PFNGLVIEWPORTPROC, "glViewport");
    LOAD_GL(p_glUseProgram, PFNGLUSEPROGRAMPROC, "glUseProgram");
    LOAD_GL(p_glGetUniformLocation, PFNGLGETUNIFORMLOCATIONPROC, "glGetUniformLocation");
    LOAD_GL(p_glUniform1i, PFNGLUNIFORM1IPROC, "glUniform1i");
    LOAD_GL(p_glGetBufferParameteriv, PFNGLGETBUFFERPARAMETERIVPROC, "glGetBufferParameteriv");
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
    GLuint programs[TERRAIN_LAYERS + 2];   /* 3 terrain layers + entity + GUI */
    GLint  sampler_locs[TERRAIN_LAYERS + 2];
    GLuint textures[8];
    GLuint samplers[3];

    /* geometry shape */
    GLsizei vertex_slice_bytes;
    GLsizei index_count;

    /* per-draw UBO allocator (DynamicUniformStorage) */
    size_t ubo_block;           /* rounded TRANSFORM_UBO_SIZE */
    size_t ubo_next_block;
    int    ubo_ring_index;

    /* state mirror to decide when the game would switch program/pipeline */
    int terrain_draws, entity_draws, gui_draws, frames;
    int emit_startup_location_queries;
} HarnessScene;

static const char *kTerrainVS =
    "#version 330 core\n"
    "layout(location=0) in vec3 aPos;\n"
    "layout(location=1) in vec2 aUV;\n"
    "layout(location=2) in uvec2 aLight;\n"
    "layout(std140) uniform DynamicTransforms { mat4 uModelView; vec4 uColor; vec3 uOffset; mat4 uTexMat; };\n"
    "uniform sampler2D Sampler0;\n"
    "out vec2 vUV;\n"
    "void main(){ vUV = aUV; gl_Position = uModelView * vec4(aPos + uOffset, 1.0); }\n";
static const char *kTerrainFS =
    "#version 330 core\n"
    "in vec2 vUV;\n"
    "uniform sampler2D Sampler0;\n"
    "out vec4 C;\n"
    "void main(){ C = texture(Sampler0, vUV); }\n";
static const char *kEntityFS =
    "#version 330 core\n"
    "in vec2 vUV;\n"
    "uniform sampler2D Sampler0;\n"
    "uniform sampler2D Sampler1;\n"
    "out vec4 C;\n"
    "void main(){ C = texture(Sampler0, vUV) * texture(Sampler1, vUV); }\n";
static const char *kGuiVS =
    "#version 330 core\n"
    "layout(location=0) in vec3 aPos;\n"
    "layout(location=1) in vec2 aUV;\n"
    "layout(std140) uniform DynamicTransforms { mat4 uModelView; vec4 uColor; vec3 uOffset; mat4 uTexMat; };\n"
    "uniform sampler2D Sampler0;\n"
    "out vec2 vUV;\n"
    "void main(){ vUV = aUV; gl_Position = uModelView * vec4(aPos, 1.0); }\n";

static GLuint compile_program(const char *vs_src, const char *fs_src)
{
    GLuint v = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(v, 1, &vs_src, NULL);
    glCompileShader(v);
    GLuint f = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(f, 1, &fs_src, NULL);
    glCompileShader(f);
    GLuint p = glCreateProgram();
    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        fprintf(stderr, "harness program link failed\n");
        return 0;
    }
    glDeleteShader(v);
    glDeleteShader(f);
    return p;
}

/* VertexArrayCache.Separate: one VAO per (format), attribute formats bound
 * once, per-VBO glBindVertexBuffer on change.  Terrain format here:
 * position(3f) + uv(2f) + light(2ub normalized->I) = 24 bytes. */
static void setup_format_vao(GLuint vao)
{
    p_glBindVertexArray(vao);
    p_glEnableVertexAttribArray(0);
    p_glEnableVertexAttribArray(1);
    p_glEnableVertexAttribArray(2);
    p_glVertexAttribFormat(0, 3, GL_FLOAT, GL_FALSE, 0);
    p_glVertexAttribFormat(1, 2, GL_FLOAT, GL_FALSE, 12);
    p_glVertexAttribIFormat(2, 2, GL_UNSIGNED_BYTE, 20);
    p_glVertexAttribBinding(0, 0);
    p_glVertexAttribBinding(1, 0);
    p_glVertexAttribBinding(2, 0);
}

static int init_scene(HarnessScene *s)
{
    memset(s, 0, sizeof(*s));
    s->terrain_draws = DEFAULT_TERRAIN_DRAWS;
    s->entity_draws = DEFAULT_ENTITY_DRAWS;
    s->gui_draws = DEFAULT_GUI_DRAWS;
    s->frames = 120;
    const char *env;
    if ((env = getenv("HARNESS_FRAMES")) != NULL) s->frames = atoi(env);
    if ((env = getenv("HARNESS_TERRAIN_DRAWS")) != NULL) s->terrain_draws = atoi(env);
    if ((env = getenv("HARNESS_ENTITY_DRAWS")) != NULL) s->entity_draws = atoi(env);
    if ((env = getenv("HARNESS_GUI_DRAWS")) != NULL) s->gui_draws = atoi(env);
    s->emit_startup_location_queries = getenv("HARNESS_STARTUP_QUERIES") != NULL;

    s->vertex_slice_bytes = 24;
    s->index_count = 6;

    glGenVertexArrays(VAO_CACHE_SIZE, s->vaos);
    for (int i = 0; i < VAO_CACHE_SIZE; i++) setup_format_vao(s->vaos[i]);

    glGenBuffers(1, &s->vertex_arena);
    p_glBindBuffer(GL_ARRAY_BUFFER, s->vertex_arena);
    p_glBufferData(GL_ARRAY_BUFFER,
                   (GLsizeiptr)s->vertex_slice_bytes * 64 * 1024, NULL, GL_STATIC_DRAW);

    glGenBuffers(VAO_CACHE_SIZE, s->ebos);
    for (int i = 0; i < VAO_CACHE_SIZE; i++) {
        p_glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s->ebos[i]);
        p_glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                       (GLsizeiptr)s->index_count * sizeof(GLuint), NULL, GL_STATIC_DRAW);
    }

    /* DynamicUniforms ring: persistent storage, TRANSFORM_UBO_SIZE blocks
     * rounded to the device uniform offset alignment (256 on Apple GL). */
    s->ubo_block = ((size_t)TRANSFORM_UBO_SIZE + UBO_BLOCK_ALIGN - 1) &
                   ~((size_t)UBO_BLOCK_ALIGN - 1);
    glGenBuffers(UBO_RING_BUFFERS, s->dynamic_ubo);
    for (int i = 0; i < UBO_RING_BUFFERS; i++) {
        p_glBindBuffer(GL_UNIFORM_BUFFER, s->dynamic_ubo[i]);
        p_glBufferData(GL_UNIFORM_BUFFER,
                       (GLsizeiptr)(s->ubo_block * (size_t)(s->terrain_draws + s->entity_draws + s->gui_draws + SKY_DRAWS + 16)),
                       NULL, GL_DYNAMIC_DRAW);
    }

    /* Render-pass targets: every pass FBO gets color+depth like the real
     * GpuTextureView FBOs (createRenderPass would otherwise fail validation). */
    glGenFramebuffers(PASS_COUNT, s->fbos);
    for (int i = 0; i < PASS_COUNT; i++) {
        GLuint color, depth;
        glGenTextures(1, &color);
        glGenTextures(1, &depth);
        p_glBindTexture(GL_TEXTURE_2D, color);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 640, 480, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        p_glBindTexture(GL_TEXTURE_2D, depth);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, 640, 480, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, NULL);
        p_glBindFramebuffer(GL_FRAMEBUFFER, s->fbos[i]);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth, 0);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            fprintf(stderr, "harness fbo %d incomplete\n", i);
            return 1;
        }
    }
    p_glBindFramebuffer(GL_FRAMEBUFFER, 0);

    for (int i = 0; i < TERRAIN_LAYERS; i++) {
        s->programs[i] = compile_program(kTerrainVS, kTerrainFS);
        s->sampler_locs[i] = p_glGetUniformLocation(s->programs[i], "Sampler0");
    }
    s->programs[TERRAIN_LAYERS] = compile_program(kTerrainVS, kEntityFS);
    s->sampler_locs[TERRAIN_LAYERS] = p_glGetUniformLocation(s->programs[TERRAIN_LAYERS], "Sampler0");
    s->programs[TERRAIN_LAYERS + 1] = compile_program(kGuiVS, kTerrainFS);
    s->sampler_locs[TERRAIN_LAYERS + 1] = p_glGetUniformLocation(s->programs[TERRAIN_LAYERS + 1], "Sampler0");

    glGenTextures(8, s->textures);
    for (int i = 0; i < 8; i++) {
        p_glBindTexture(GL_TEXTURE_2D, s->textures[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 64, 64, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        /* Minecraft always sets explicit filters on its textures; without
         * this the default mipmap filter makes the texture sampling-
         * incomplete and MGL substitutes its fallback texture per draw. */
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }
    glGenSamplers(3, s->samplers);
    /* RenderPipeline sampler descriptions always set explicit filters; a
     * default sampler object (mipmap min filter) would make every sampled
     * texture incomplete and push MGL into its fallback path. */
    for (int i = 0; i < 3; i++) {
        glSamplerParameteri(s->samplers[i], GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glSamplerParameteri(s->samplers[i], GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glSamplerParameteri(s->samplers[i], GL_TEXTURE_WRAP_S, GL_REPEAT);
        glSamplerParameteri(s->samplers[i], GL_TEXTURE_WRAP_T, GL_REPEAT);
    }
    return 0;
}

/* GlRenderPass.setUniform + trySetup's per-draw UBO slice binding. */
static void bind_draw_ubo_slice(HarnessScene *s, GLuint program_binding)
{
    size_t offset = s->ubo_next_block * s->ubo_block;
    p_glBindBufferRange(GL_UNIFORM_BUFFER, program_binding,
                        s->dynamic_ubo[s->ubo_ring_index],
                        (GLintptr)offset, (GLsizeiptr)s->ubo_block);
    s->ubo_next_block++;
}

/* GlCommandEncoder.trySetup sampler block: runs for every sampler uniform on
 * every trySetup, unconditionally (bind + sampler + min/max LOD params).
 * HARNESS_SKIP_LOD_PARAMS=1 drops only the two glTexParameteri calls to
 * isolate their MGL-side cost; HARNESS_SKIP_SAMPLER_BLOCK=1 drops the whole
 * block (both are measurement knobs, vanilla keeps them). */
static int g_skip_lod_params;
static int g_skip_sampler_block;
/* Present every frame like the real game (default on).  Disabling is only
 * for pure submit stress; unbounded in-flight commits then accumulate. */
static int g_present_per_frame = 1;
static GLFWwindow *g_window;

static void try_setup_samplers(int sampler_count, GLuint texture, GLuint sampler)
{
    if (g_skip_sampler_block) return;
    for (int u = 0; u < sampler_count; u++) {
        p_glActiveTexture(GL_TEXTURE0 + u);
        p_glBindTexture(GL_TEXTURE_2D, texture);
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

/* createRenderPass: FBO bind (+ optional clear) + viewport. */
static void begin_pass(HarnessScene *s, int pass, int width, int height)
{
    p_glBindFramebuffer(GL_FRAMEBUFFER, s->fbos[pass]);
    if (pass < CLEAR_PASSES) {
        glDisable(GL_SCISSOR_TEST);
        glDepthMask(GL_TRUE);
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        glClearColor(0.5f, 0.7f, 1.0f, 1.0f);
        glClearDepth(1.0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }
    p_glViewport(0, 0, width, height);
}

/* VertexArrayCache.bindVertexArray on the Separate path: VAO cache hit +
 * glBindVertexBuffer when the VBO changes (here: rarely — slot cycling). */
static void bind_vertex_source(HarnessScene *s, int slot, int *last_slot)
{
    p_glBindVertexArray(s->vaos[slot % VAO_CACHE_SIZE]);
    if (slot != *last_slot) {
        p_glBindVertexBuffer(0, s->vertex_arena,
                             (GLintptr)((size_t)(slot % 64) * s->vertex_slice_bytes),
                             s->vertex_slice_bytes);
        *last_slot = slot;
    }
}

static void draw_frame(HarnessScene *s, int frame)
{
    int width = 640, height = 480;
    int last_vbo_slot = -1;
    int last_program = -1;
    int last_pipeline_key = -1;

    /* DynamicUniforms.reset(): rotate ring, reset allocator. */
    s->ubo_ring_index = frame % UBO_RING_BUFFERS;
    s->ubo_next_block = 0;

    /* ---- pass 0: sky (array draws, no depth, own program) ---- */
    begin_pass(s, 0, width, height);
    apply_pipeline_state(0, 1, 0);
    last_pipeline_key = 0;
    if (last_program != 0) {
        p_glUseProgram(s->programs[0]);
        p_glUniform1i(s->sampler_locs[0], 0);
        last_program = 0;
    }
    for (int i = 0; i < SKY_DRAWS; i++) {
        bind_vertex_source(s, i, &last_vbo_slot);
        try_setup_samplers(1, s->textures[0], s->samplers[0]);
        bind_draw_ubo_slice(s, 1);
        p_glDrawArrays(GL_TRIANGLES, 0, 6);
    }

    /* ---- pass 1: terrain (3 layer programs, indexed, atlas texture) ---- */
    begin_pass(s, 1, width, height);
    apply_pipeline_state(1, 1, 0);
    last_pipeline_key = 1;
    for (int i = 0; i < s->terrain_draws; i++) {
        int layer = i % TERRAIN_LAYERS;
        if (last_program != layer) {
            p_glUseProgram(s->programs[layer]);
            p_glUniform1i(s->sampler_locs[layer], 0);
            last_program = layer;
        }
        bind_vertex_source(s, i, &last_vbo_slot);
        p_glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s->ebos[i % VAO_CACHE_SIZE]);
        try_setup_samplers(1, s->textures[0], s->samplers[0]);
        bind_draw_ubo_slice(s, 1);
        p_glDrawElements(GL_TRIANGLES, s->index_count, GL_UNSIGNED_INT, (void *)0);
    }

    /* ---- pass 2: entities (per-draw texture alternation, 2 samplers) ---- */
    begin_pass(s, 2, width, height);
    if (last_pipeline_key != 2) {
        apply_pipeline_state(1, 0, 0);
        last_pipeline_key = 2;
    }
    for (int i = 0; i < s->entity_draws; i++) {
        if (last_program != TERRAIN_LAYERS) {
            p_glUseProgram(s->programs[TERRAIN_LAYERS]);
            p_glUniform1i(s->sampler_locs[TERRAIN_LAYERS], 0);
            last_program = TERRAIN_LAYERS;
        }
        bind_vertex_source(s, i, &last_vbo_slot);
        p_glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s->ebos[i % VAO_CACHE_SIZE]);
        try_setup_samplers(2, s->textures[1 + (i % ENTITY_TEXTURE_CYCLE)], s->samplers[1]);
        bind_draw_ubo_slice(s, 1);
        p_glDrawElements(GL_TRIANGLES, s->index_count, GL_UNSIGNED_INT, (void *)0);
    }

    /* ---- pass 3: translucent (blend on) ---- */
    begin_pass(s, 3, width, height);
    apply_pipeline_state(1, 0, 1);
    last_pipeline_key = 3;
    for (int i = 0; i < s->terrain_draws / 8; i++) {
        int layer = i % TERRAIN_LAYERS;
        if (last_program != layer) {
            p_glUseProgram(s->programs[layer]);
            p_glUniform1i(s->sampler_locs[layer], 0);
            last_program = layer;
        }
        bind_vertex_source(s, i, &last_vbo_slot);
        p_glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s->ebos[i % VAO_CACHE_SIZE]);
        try_setup_samplers(1, s->textures[0], s->samplers[2]);
        bind_draw_ubo_slice(s, 1);
        p_glDrawElements(GL_TRIANGLES, s->index_count, GL_UNSIGNED_INT, (void *)0);
    }

    /* ---- pass 4: clouds + weather (small, array draws, blend) ---- */
    begin_pass(s, 4, width, height);
    apply_pipeline_state(0, 0, 1);
    last_pipeline_key = 4;
    for (int i = 0; i < 32; i++) {
        bind_vertex_source(s, i, &last_vbo_slot);
        try_setup_samplers(1, s->textures[7], s->samplers[2]);
        bind_draw_ubo_slice(s, 1);
        p_glDrawArrays(GL_TRIANGLES, 0, 6);
    }

    /* ---- pass 5: GUI (blend, texture per 8 draws, scissor sometimes) ---- */
    begin_pass(s, 5, width, height);
    apply_pipeline_state(0, 0, 1);
    last_pipeline_key = 5;
    if (last_program != TERRAIN_LAYERS + 1) {
        p_glUseProgram(s->programs[TERRAIN_LAYERS + 1]);
        p_glUniform1i(s->sampler_locs[TERRAIN_LAYERS + 1], 0);
        last_program = TERRAIN_LAYERS + 1;
    }
    for (int i = 0; i < s->gui_draws; i++) {
        if ((i % 16) == 0) {
            glEnable(GL_SCISSOR_TEST);
            glScissor((i * 8) % width, 0, 128, 64);
        }
        bind_vertex_source(s, 0, &last_vbo_slot);
        p_glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s->ebos[i % VAO_CACHE_SIZE]);
        try_setup_samplers(1, s->textures[(i / GUI_TEXTURE_GROUP) % 4], s->samplers[0]);
        bind_draw_ubo_slice(s, 1);
        p_glDrawElements(GL_TRIANGLES, s->index_count, GL_UNSIGNED_INT, (void *)0);
    }
    glDisable(GL_SCISSOR_TEST);

    /* GlCommandEncoder.finishRenderPass */
    p_glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IOLBF, 0);
    int json = 0;
    const char *json_path = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--json") == 0 && i + 1 < argc) {
            json = 1;
            json_path = argv[++i];
        }
    }

    g_probe_counter = (unsigned long *)dlsym(RTLD_DEFAULT, "mgl_vm_probe_count");
    g_skip_lod_params = getenv("HARNESS_SKIP_LOD_PARAMS") != NULL;
    g_skip_sampler_block = getenv("HARNESS_SKIP_SAMPLER_BLOCK") != NULL;

    if (!glfwInit()) {
        fprintf(stderr, "glfwInit failed\n");
        return 1;
    }
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow *win = glfwCreateWindow(640, 480, "mc_offline_harness", NULL, NULL);
    g_window = win;
    if (!win) return 1;
    glfwMakeContextCurrent(win);
    if (load_gl() != 0) return 1;

    HarnessScene scene;
    if (init_scene(&scene) != 0) return 1;

    int warmup = 10;
    const char *env = getenv("HARNESS_WARMUP");
    if (env) warmup = atoi(env);

    /* Optional startup phase (shaderpack-style location resolution). */
    unsigned long probes_before_startup = probe_count();
    double startup_wall_0 = now_ns(CLOCK_MONOTONIC);
    if (scene.emit_startup_location_queries) {
        /* already ran inside init_scene; re-create it here? — it ran there;
         * measure by delta between init end and now is meaningless, so
         * rerun a second 24-uniform program here for a clean window. */
        char fs_src[2048];
        size_t off = 0;
        off += (size_t)snprintf(fs_src + off, sizeof fs_src - off, "#version 330 core\nout vec4 C;\n");
        for (int i = 0; i < 24; i++)
            off += (size_t)snprintf(fs_src + off, sizeof fs_src - off, "uniform vec4 uV%d;\n", i);
        off += (size_t)snprintf(fs_src + off, sizeof fs_src - off, "void main(){C=uV0;}\n");
        const char *fsrc = fs_src;
        GLuint f = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(f, 1, &fsrc, NULL);
        glCompileShader(f);
        GLuint p = glCreateProgram();
        const char *vs1 = "#version 330 core\nvoid main(){gl_Position=vec4(0.0);}\n";
        GLuint v = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(v, 1, &vs1, NULL);
        glCompileShader(v);
        glAttachShader(p, v);
        glAttachShader(p, f);
        glLinkProgram(p);
        p_glUseProgram(p);
        for (int rep = 0; rep < 300; rep++) {
            for (int i = 0; i < 24; i++) {
                char name[16];
                snprintf(name, sizeof name, "uV%d", i);
                (void)p_glGetUniformLocation(p, name);
            }
        }
        p_glUseProgram(0);
    }
    double startup_wall_ms = (now_ns(CLOCK_MONOTONIC) - startup_wall_0) / 1e6;
    unsigned long startup_probes = probe_count() - probes_before_startup;

    for (int i = 0; i < warmup; i++) draw_frame(&scene, i);
    glFinish();

    int frames = scene.frames;
    double *wall_ms = malloc((size_t)frames * sizeof(double));
    double *user_ms = malloc((size_t)frames * sizeof(double));
    double *sys_ms = malloc((size_t)frames * sizeof(double));
    double *swap_ms = malloc((size_t)frames * sizeof(double));
    if (!wall_ms || !user_ms || !sys_ms || !swap_ms) return 1;
    {
        const char *pf = getenv("HARNESS_NO_PRESENT");
        g_present_per_frame = (pf && pf[0] == '1') ? 0 : 1;
    }

    unsigned long probes_before = probe_count();
    struct rusage r0, r1;
    getrusage(RUSAGE_SELF, &r0);
    double wall0 = now_ns(CLOCK_MONOTONIC);

    for (int f = 0; f < frames; f++) {
        struct rusage rf0, rf1;
        getrusage(RUSAGE_SELF, &rf0);
        double f0 = now_ns(CLOCK_MONOTONIC);
        draw_frame(&scene, f);
        double f1 = now_ns(CLOCK_MONOTONIC);
        /* Present like the real game (RenderSystem/Glfw swap every frame).
         * Without a per-frame present MGL's in-flight commits and CoW
         * snapshots are never drained and accumulate without bound — at
         * HARNESS_FRAMES=3000 that was tens of GB.  Present keeps in-flight
         * work bounded at one frame, exactly as the game does.  Swap time is
         * reported separately and not folded into the submit stats. */
        double s0 = now_ns(CLOCK_MONOTONIC);
        if (g_present_per_frame) glfwSwapBuffers(g_window);
        double s1 = now_ns(CLOCK_MONOTONIC);
        getrusage(RUSAGE_SELF, &rf1);
        wall_ms[f] = (f1 - f0) / 1e6;
        swap_ms[f] = (s1 - s0) / 1e6;
        user_ms[f] = ((double)(rf1.ru_utime.tv_sec - rf0.ru_utime.tv_sec) * 1e6 +
                      (double)(rf1.ru_utime.tv_usec - rf0.ru_utime.tv_usec)) / 1e3;
        sys_ms[f] = ((double)(rf1.ru_stime.tv_sec - rf0.ru_stime.tv_sec) * 1e6 +
                     (double)(rf1.ru_stime.tv_usec - rf0.ru_stime.tv_usec)) / 1e3;
    }

    getrusage(RUSAGE_SELF, &r1);
    double wall_total_ms = (now_ns(CLOCK_MONOTONIC) - wall0) / 1e6;
    unsigned long probes_total = probe_count() - probes_before;

    FrameStats st;
    summarize(&st, wall_ms, user_ms, sys_ms, frames);
    double *swap_sorted = malloc((size_t)frames * sizeof(double));
    memcpy(swap_sorted, swap_ms, (size_t)frames * sizeof(double));
    qsort(swap_sorted, (size_t)frames, sizeof(double), cmp_double);
    double swap_p50 = swap_sorted[(int)(frames * 0.50)];
    free(swap_sorted);

    double total_user_ms = ((double)(r1.ru_utime.tv_sec - r0.ru_utime.tv_sec) * 1e6 +
                            (double)(r1.ru_utime.tv_usec - r0.ru_utime.tv_usec)) / 1e3;
    double total_sys_ms = ((double)(r1.ru_stime.tv_sec - r0.ru_stime.tv_sec) * 1e6 +
                           (double)(r1.ru_stime.tv_usec - r0.ru_stime.tv_usec)) / 1e3;

    int draws_per_frame = scene.terrain_draws + scene.entity_draws + scene.gui_draws + SKY_DRAWS +
                          scene.terrain_draws / 8 + 32;
    printf("backend=mgl frames=%d warmup=%d draws_per_frame=%d\n",
           frames, warmup, draws_per_frame);
    printf("frame_wall_ms=%.3f frame_cpu_ms=%.3f frame_user_ms=%.3f frame_sys_ms=%.3f p50_ms=%.3f p95_ms=%.3f swap_p50_ms=%.3f present=%d\n",
           st.wall_per_frame_ms, st.cpu_per_frame_ms, st.user_per_frame_ms, st.sys_per_frame_ms,
           st.p50_ms, st.p95_ms, swap_p50, g_present_per_frame);
    printf("window_wall_ms=%.1f window_user_ms=%.1f window_sys_ms=%.1f\n",
           wall_total_ms, total_user_ms, total_sys_ms);
    printf("probes_available=%d probes_per_frame=%.1f startup_queries_ms=%.3f startup_probes=%lu\n",
           g_probe_counter != NULL,
           g_probe_counter ? (double)probes_total / frames : 0.0,
           startup_wall_ms, startup_probes);

    if (json && json_path) {
        FILE *f = fopen(json_path, "w");
        if (f) {
            fprintf(f,
                    "{\"schema_version\":1,\"tool\":\"mc_offline_harness\",\"frames\":%d,"
                    "\"warmup\":%d,\"draws_per_frame\":%d,"
                    "\"frame_wall_ms\":%.3f,\"frame_cpu_ms\":%.3f,\"frame_user_ms\":%.3f,"
                    "\"frame_sys_ms\":%.3f,\"p50_ms\":%.3f,\"p95_ms\":%.3f,"
                    "\"window_user_ms\":%.1f,\"window_sys_ms\":%.1f,"
                    "\"probes_available\":%s,\"probes_per_frame\":%.1f,"
                    "\"startup_queries_ms\":%.3f,\"startup_probes\":%lu}\n",
                    frames, warmup, draws_per_frame,
                    st.wall_per_frame_ms, st.cpu_per_frame_ms, st.user_per_frame_ms,
                    st.sys_per_frame_ms, st.p50_ms, st.p95_ms,
                    total_user_ms, total_sys_ms,
                    g_probe_counter ? "true" : "false",
                    g_probe_counter ? (double)probes_total / frames : 0.0,
                    startup_wall_ms, startup_probes);
            fclose(f);
        }
    }

    glfwDestroyWindow(win);
    glfwTerminate();
    free(wall_ms);
    free(user_ms);
    free(sys_ms);
    free(swap_ms);
    return 0;
}
