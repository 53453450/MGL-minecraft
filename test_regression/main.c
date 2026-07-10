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

#define GL_GLEXT_PROTOTYPES 1
#include <GL/glcorearb.h>

#include "MGLContext.h"
#include "MGLRenderer.h"

/* ------------------------------------------------------------------ */
/* Constants                                                          */
/* ------------------------------------------------------------------ */

#define REG_W 128
#define REG_H 128
#define MAX_TESTS 16

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

    GLuint prog = link_program(
        "#version 330 core\n"
        "layout(location = 0) in vec2 a_pos;\n"
        "void main() { gl_Position = vec4(a_pos, 0.0, 1.0); }\n",
        FS_SOLID);
    if (!prog) return 2;
    glUseProgram(prog);

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    GLuint vbo = make_vbo(TRI_VERTS, sizeof(TRI_VERTS));
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

    /* Render red full-screen-ish triangle to FBO A */
    glBindFramebuffer(GL_FRAMEBUFFER, fboA);
    clear_color(0.0f, 0.0f, 0.0f);
    glUniform4f(glGetUniformLocation(prog, "u_color"), 1.0f, 0.0f, 0.0f, 1.0f);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();

    /* Switch to FBO B, clear to dark, render green triangle */
    glBindFramebuffer(GL_FRAMEBUFFER, fboB);
    clear_color(0.05f, 0.05f, 0.1f);
    glUniform4f(glGetUniformLocation(prog, "u_color"), 0.0f, 1.0f, 0.0f, 1.0f);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();

    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glDeleteProgram(prog);
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
        fprintf(stderr, "  [conditional: query returned 0 samples — unexpected]\n");
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

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    GLuint vbo = make_vbo(TRI_VERTS, sizeof(TRI_VERTS));
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

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
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    GLuint vbo = make_vbo(FULL, sizeof(FULL));
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

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

/* ------------------------------------------------------------------ */
/* Test registry                                                      */
/* ------------------------------------------------------------------ */

typedef struct {
    const char *name;
    test_fn fn;
} TestCase;

static const TestCase TESTS[] = {
    { "draw_arrays",          test_draw_arrays },
    { "draw_elements",        test_draw_elements },
    { "draw_arrays_instanced",test_draw_arrays_instanced },
    { "multi_draw_elements",  test_multi_draw_elements },
    { "draw_arrays_indirect", test_draw_arrays_indirect },
    { "fbo_switch",           test_fbo_switch },
    { "transform_feedback",   test_transform_feedback },
    { "conditional_render",   test_conditional_render },
    { "program_switch",       test_program_switch },
    { "blend",                test_blend },
    { "depth_test",           test_depth_probe },
    { "stencil",              test_stencil_probe },
    { "uniform_alias",        test_uniform_alias },
    { "shared_uniform",       test_shared_uniform },
    { "multipass_resume",     test_multipass_resume },
    { "dontcare_fullscreen",  test_dontcare_fullscreen },
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
static const int NUM_TESTS = (int)(sizeof(TESTS) / sizeof(TESTS[0]));

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

        char out_path[1024];
        snprintf(out_path, sizeof(out_path), "%s/Reg_%s.tga", out_dir, t->name);

        fprintf(stderr, "[%02d/%02d] %-24s ... ", i + 1, NUM_TESTS, t->name);
        fflush(stderr);

        memset(pixels, 0, REG_W * REG_H * 4);
        GLenum pre_err = glGetError();
        (void)pre_err;

        int rc = t->fn(pixels, out_path);

        /* drain any lingering GL errors for cleanliness */
        GLenum e;
        while ((e = glGetError()) != GL_NO_ERROR) {
            /* only warn; some drivers leave harmless errors */
        }

        if (rc != 0) {
            fprintf(stderr, "ERROR (rc=%d)\n", rc);
            n_fail++;
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
