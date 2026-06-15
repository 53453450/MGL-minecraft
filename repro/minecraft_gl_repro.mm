#include <cassert>
#include <cerrno>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>

#define GL_GLEXT_PROTOTYPES 1
#include <GL/glcorearb.h>

#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3native.h>

extern "C" {
#include "MGLContext.h"
}
#include "MGLRenderer.h"

static const int kWidth = 256;
static const int kHeight = 256;
static const char *kDefaultOutputDir = "mgl-repro-output";

struct Pixel {
    uint8_t r, g, b, a;
};

static void glfwError(int error, const char *description)
{
    fprintf(stderr, "GLFW error %d: %s\n", error, description);
}

static const char *outputDir()
{
    const char *env = getenv("MGL_REPRO_OUTPUT_DIR");
    return env && env[0] ? env : kDefaultOutputDir;
}

static void ensureOutputDir()
{
    const char *dir = outputDir();
    char path[1024];
    snprintf(path, sizeof(path), "%s", dir);

    size_t len = strlen(path);
    while (len > 1 && path[len - 1] == '/') {
        path[--len] = '\0';
    }

    for (char *p = path + 1; *p; ++p) {
        if (*p != '/') {
            continue;
        }
        *p = '\0';
        if (mkdir(path, 0755) != 0 && errno != EEXIST) {
            fprintf(stderr, "failed to create output directory %s: errno=%d\n", path, errno);
            exit(2);
        }
        *p = '/';
    }

    if (mkdir(path, 0755) != 0 && errno != EEXIST) {
        fprintf(stderr, "failed to create output directory %s: errno=%d\n", path, errno);
        exit(2);
    }
}

static void writePpm(const char *name, int width, int height, const std::vector<Pixel> &pixels)
{
    ensureOutputDir();

    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.ppm", outputDir(), name);

    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "failed to open %s\n", path);
        exit(2);
    }

    fprintf(fp, "P6\n%d %d\n255\n", width, height);
    for (int y = height - 1; y >= 0; --y) {
        for (int x = 0; x < width; ++x) {
            const Pixel &p = pixels[y * width + x];
            fputc(p.r, fp);
            fputc(p.g, fp);
            fputc(p.b, fp);
        }
    }
    fclose(fp);
    fprintf(stderr, "wrote %s\n", path);
}

static void failGl(const char *where, GLenum err)
{
    fprintf(stderr, "GL error at %s: 0x%x\n", where, err);
    exit(2);
}

static void checkGl(const char *where)
{
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        failGl(where, err);
    }
}

static GLuint compileShader(GLenum type, const char *src)
{
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    GLint ok = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        GLint len = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);
        std::vector<char> log(len > 1 ? len : 1);
        glGetShaderInfoLog(shader, (GLsizei)log.size(), nullptr, log.data());
        fprintf(stderr, "shader compile failed:\n%s\n", log.data());
        exit(2);
    }
    return shader;
}

static GLuint linkProgram(int count, ...)
{
    GLuint program = glCreateProgram();

    va_list args;
    va_start(args, count);
    std::vector<GLuint> shaders;
    for (int i = 0; i < count; ++i) {
        GLenum type = va_arg(args, GLenum);
        const char *src = va_arg(args, const char *);
        GLuint shader = compileShader(type, src);
        glAttachShader(program, shader);
        shaders.push_back(shader);
    }
    va_end(args);

    glLinkProgram(program);

    GLint ok = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (!ok) {
        GLint len = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &len);
        std::vector<char> log(len > 1 ? len : 1);
        glGetProgramInfoLog(program, (GLsizei)log.size(), nullptr, log.data());
        fprintf(stderr, "program link failed:\n%s\n", log.data());
        exit(2);
    }

    for (GLuint shader : shaders) {
        glDeleteShader(shader);
    }
    return program;
}

static GLFWwindow *createMglWindow(const char *title)
{
    glfwSetErrorCallback(glfwError);
    if (!glfwInit()) {
        fprintf(stderr, "glfwInit failed\n");
        exit(2);
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_COCOA_RETINA_FRAMEBUFFER, GL_TRUE);
    glfwWindowHint(GLFW_CONTEXT_CREATION_API, GLFW_NATIVE_CONTEXT_API);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_DEPTH_BITS, 32);

    GLFWwindow *window = glfwCreateWindow(kWidth / 2, kHeight / 2, title, nullptr, nullptr);
    if (!window) {
        fprintf(stderr, "glfwCreateWindow failed\n");
        exit(2);
    }

    GLMContext ctx = createGLMContext(GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV,
                                      GL_DEPTH_COMPONENT, GL_FLOAT, 0, 0);
    void *renderer = CppCreateMGLRendererFromContextAndBindToWindow(ctx, (__bridge void *)glfwGetCocoaWindow(window));
    if (!renderer) {
        fprintf(stderr, "CppCreateMGLRendererFromContextAndBindToWindow failed\n");
        exit(2);
    }

    MGLsetCurrentContext(ctx);
    glfwSetWindowUserPointer(window, ctx);
    glfwSwapInterval(0);

    return window;
}

static GLuint makeTexture2D(int width, int height, GLint internalFormat, GLenum format, GLenum type,
                            const void *pixels = nullptr)
{
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, format, type, pixels);
    glBindTexture(GL_TEXTURE_2D, 0);
    return tex;
}

static GLuint makeFbo(GLuint colorTex, GLuint depthTex)
{
    GLuint fbo = 0;
    glCreateFramebuffers(1, &fbo);
    glNamedFramebufferTexture(fbo, GL_COLOR_ATTACHMENT0, colorTex, 0);
    if (depthTex) {
        glNamedFramebufferTexture(fbo, GL_DEPTH_ATTACHMENT, depthTex, 0);
    }
    glNamedFramebufferDrawBuffer(fbo, GL_COLOR_ATTACHMENT0);
    glNamedFramebufferReadBuffer(fbo, GL_COLOR_ATTACHMENT0);

    GLenum status = glCheckNamedFramebufferStatus(fbo, GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "FBO incomplete: 0x%x\n", status);
        exit(2);
    }
    return fbo;
}

static std::vector<Pixel> readBack(int width, int height)
{
    std::vector<Pixel> pixels((size_t)width * (size_t)height);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    checkGl("glReadPixels");
    return pixels;
}

static Pixel sample(const std::vector<Pixel> &pixels, int width, int x, int y)
{
    return pixels[(size_t)y * (size_t)width + (size_t)x];
}

static bool expectDominant(const char *label, Pixel p, char channel)
{
    uint8_t v = channel == 'r' ? p.r : channel == 'g' ? p.g : p.b;
    uint8_t a = channel == 'r' ? p.g : p.r;
    uint8_t b = channel == 'b' ? p.g : p.b;
    bool ok = v > a + 20 && v > b + 20 && p.a > 200;
    fprintf(stderr, "%s rgba=(%u,%u,%u,%u) dominant=%c => %s\n",
            label, p.r, p.g, p.b, p.a, channel, ok ? "pass" : "FAIL");
    return ok;
}

static GLuint makeSampler(GLint minFilter = GL_LINEAR, GLint magFilter = GL_LINEAR)
{
    GLuint sampler = 0;
    glCreateSamplers(1, &sampler);
    glSamplerParameteri(sampler, GL_TEXTURE_MIN_FILTER, minFilter);
    glSamplerParameteri(sampler, GL_TEXTURE_MAG_FILTER, magFilter);
    glSamplerParameteri(sampler, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glSamplerParameteri(sampler, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return sampler;
}

static GLuint makeCheckerTexture(int width, int height, uint8_t r0, uint8_t g0, uint8_t b0,
                                 uint8_t r1, uint8_t g1, uint8_t b1)
{
    std::vector<uint8_t> data((size_t)width * (size_t)height * 4);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            bool hi = (((x / 16) ^ (y / 16)) & 1) != 0;
            size_t o = ((size_t)y * (size_t)width + (size_t)x) * 4;
            data[o + 0] = hi ? r1 : r0;
            data[o + 1] = hi ? g1 : g0;
            data[o + 2] = hi ? b1 : b0;
            data[o + 3] = 255;
        }
    }
    return makeTexture2D(width, height, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, data.data());
}

static GLuint makeCubeTexture()
{
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_CUBE_MAP, tex);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    for (int face = 0; face < 6; ++face) {
        uint8_t data[16 * 16 * 4];
        for (int i = 0; i < 16 * 16; ++i) {
            data[i * 4 + 0] = (uint8_t)(25 + face * 20);
            data[i * 4 + 1] = (uint8_t)(45 + face * 10);
            data[i * 4 + 2] = (uint8_t)(95 + face * 18);
            data[i * 4 + 3] = 255;
        }
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + face, 0, GL_RGBA8, 16, 16, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, data);
    }
    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
    return tex;
}

static GLuint makePersistentBuffer(GLsizeiptr size)
{
    GLuint buffer = 0;
    glCreateBuffers(1, &buffer);
    glNamedBufferStorage(buffer, size, nullptr,
                         GL_MAP_WRITE_BIT | GL_DYNAMIC_STORAGE_BIT);
    return buffer;
}

static void writeMappedRange(GLuint buffer, GLintptr offset, GLsizeiptr size, const void *src)
{
    void *dst = glMapNamedBufferRange(buffer, offset, size,
                                      GL_MAP_WRITE_BIT | GL_MAP_FLUSH_EXPLICIT_BIT);
    if (!dst) {
        fprintf(stderr, "glMapNamedBufferRange failed for buffer %u\n", buffer);
        exit(2);
    }
    memcpy(dst, src, (size_t)size);
    glFlushMappedNamedBufferRange(buffer, 0, size);
    glUnmapNamedBuffer(buffer);
}

static void waitOnFence()
{
    GLsync sync = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    if (sync) {
        glClientWaitSync(sync, 0, 1000000000ull);
        glDeleteSync(sync);
    }
}

struct TailVertex {
    float x, y, u, v, r, g;
};

struct TailDualVertex {
    float x, y, u, v, lu, lv, mix;
};

static void fillTailVertices(std::vector<TailVertex> &vertices)
{
    const TailVertex quad[4] = {
        {-0.90f, -0.90f, 0.0f, 0.0f, 0.95f, 0.45f},
        { 0.90f, -0.90f, 1.0f, 0.0f, 0.25f, 0.75f},
        { 0.90f,  0.90f, 1.0f, 1.0f, 0.80f, 0.85f},
        {-0.90f,  0.90f, 0.0f, 1.0f, 0.35f, 0.35f},
    };
    for (size_t i = 0; i < vertices.size(); ++i) {
        TailVertex v = quad[i & 3];
        float cell = (float)((i / 4) % 17);
        float row = (float)((i / 68) % 9);
        v.x = v.x * 0.09f + -0.86f + cell * 0.105f;
        v.y = v.y * 0.12f + -0.82f + row * 0.205f;
        v.r = 0.25f + 0.65f * ((int)i % 5) / 4.0f;
        v.g = 0.20f + 0.70f * ((int)i % 7) / 6.0f;
        vertices[i] = v;
    }
}

static void fillTailDualVertices(std::vector<TailDualVertex> &vertices)
{
    for (size_t i = 0; i < vertices.size(); ++i) {
        float a = (float)(i % 30) / 29.0f;
        float b = (float)((i / 30) % 20) / 19.0f;
        vertices[i] = {-0.95f + a * 1.90f, -0.95f + b * 1.90f,
                       a, b, 1.0f - a, b, 0.25f + 0.65f * a};
    }
}

static GLuint makeTailVao(GLuint vbo, GLuint ebo)
{
    GLuint vao = 0;
    glCreateVertexArrays(1, &vao);
    glVertexArrayVertexBuffer(vao, 0, vbo, 0, sizeof(TailVertex));
    glEnableVertexArrayAttrib(vao, 0);
    glEnableVertexArrayAttrib(vao, 1);
    glEnableVertexArrayAttrib(vao, 2);
    glVertexArrayAttribFormat(vao, 0, 2, GL_FLOAT, GL_FALSE, offsetof(TailVertex, x));
    glVertexArrayAttribFormat(vao, 1, 2, GL_FLOAT, GL_FALSE, offsetof(TailVertex, u));
    glVertexArrayAttribFormat(vao, 2, 2, GL_FLOAT, GL_FALSE, offsetof(TailVertex, r));
    glVertexArrayAttribBinding(vao, 0, 0);
    glVertexArrayAttribBinding(vao, 1, 0);
    glVertexArrayAttribBinding(vao, 2, 0);
    glVertexArrayElementBuffer(vao, ebo);
    return vao;
}

static GLuint makeTailDualVao(GLuint vbo, GLuint ebo)
{
    GLuint vao = 0;
    glCreateVertexArrays(1, &vao);
    glVertexArrayVertexBuffer(vao, 0, vbo, 0, sizeof(TailDualVertex));
    glEnableVertexArrayAttrib(vao, 0);
    glEnableVertexArrayAttrib(vao, 1);
    glEnableVertexArrayAttrib(vao, 2);
    glEnableVertexArrayAttrib(vao, 3);
    glVertexArrayAttribFormat(vao, 0, 2, GL_FLOAT, GL_FALSE, offsetof(TailDualVertex, x));
    glVertexArrayAttribFormat(vao, 1, 2, GL_FLOAT, GL_FALSE, offsetof(TailDualVertex, u));
    glVertexArrayAttribFormat(vao, 2, 2, GL_FLOAT, GL_FALSE, offsetof(TailDualVertex, lu));
    glVertexArrayAttribFormat(vao, 3, 1, GL_FLOAT, GL_FALSE, offsetof(TailDualVertex, mix));
    glVertexArrayAttribBinding(vao, 0, 0);
    glVertexArrayAttribBinding(vao, 1, 0);
    glVertexArrayAttribBinding(vao, 2, 0);
    glVertexArrayAttribBinding(vao, 3, 0);
    glVertexArrayElementBuffer(vao, ebo);
    return vao;
}

static GLuint makeFullscreenVao()
{
    GLuint vao = 0;
    glCreateVertexArrays(1, &vao);
    return vao;
}

static int runCloudTboVertexId()
{
    createMglWindow("cloud-tbo-vertexid");

    static const char *vs = R"GLSL(
#version 330

layout(std140) uniform CloudInfo {
    vec4 CloudColor;
    vec3 CloudOffset;
    vec3 CellSize;
};

uniform isamplerBuffer CloudFaces;

out vec4 vertexColor;

const int FLAG_MASK_DIR = 7;
const int FLAG_EXTRA_Z = 1 << 6;
const int FLAG_EXTRA_X = 1 << 7;

const vec2 vertices[4] = vec2[](
    vec2(0.0, 0.0),
    vec2(1.0, 0.0),
    vec2(1.0, 1.0),
    vec2(0.0, 1.0)
);

void main() {
    int quadVertex = gl_VertexID % 4;
    int index = (gl_VertexID / 4) * 3;

    int cellX = texelFetch(CloudFaces, index).r;
    int cellZ = texelFetch(CloudFaces, index + 1).r;
    int dirAndFlags = texelFetch(CloudFaces, index + 2).r;
    int direction = dirAndFlags & FLAG_MASK_DIR;

    cellX = (cellX << 1) | ((dirAndFlags & FLAG_EXTRA_X) >> 7);
    cellZ = (cellZ << 1) | ((dirAndFlags & FLAG_EXTRA_Z) >> 6);

    vec2 local = vertices[quadVertex];
    vec2 p = vec2(-0.96, -0.92) + (vec2(cellX, cellZ) + local) * CellSize.xz + CloudOffset.xz;
    gl_Position = vec4(p, 0.0, 1.0);

    float stripe = float((cellX ^ cellZ ^ direction) & 1);
    vertexColor = mix(vec4(0.08, 0.35, 0.95, 1.0),
                      vec4(0.95, 0.85, 0.12, 1.0),
                      stripe) * CloudColor;
}
)GLSL";

    static const char *fs = R"GLSL(
#version 330
in vec4 vertexColor;
out vec4 fragColor;
void main() {
    fragColor = vertexColor;
}
)GLSL";

    GLuint program = linkProgram(2, GL_VERTEX_SHADER, vs, GL_FRAGMENT_SHADER, fs);
    glUseProgram(program);

    const GLuint quadCount = 9814;
    const GLuint indexCount = quadCount * 6;
    std::vector<int8_t> cloudFaces((size_t)quadCount * 3);
    std::vector<uint32_t> indices(indexCount);

    for (GLuint q = 0; q < quadCount; ++q) {
        int x = (int)(q % 128);
        int z = (int)((q / 128) % 80);
        int flags = (q & 7);
        if (x & 1) flags |= 1 << 7;
        if (z & 1) flags |= 1 << 6;
        cloudFaces[q * 3 + 0] = (int8_t)(x >> 1);
        cloudFaces[q * 3 + 1] = (int8_t)(z >> 1);
        cloudFaces[q * 3 + 2] = (int8_t)flags;

        uint32_t base = q * 4;
        indices[q * 6 + 0] = base + 0;
        indices[q * 6 + 1] = base + 1;
        indices[q * 6 + 2] = base + 2;
        indices[q * 6 + 3] = base + 0;
        indices[q * 6 + 4] = base + 2;
        indices[q * 6 + 5] = base + 3;
    }

    GLuint faceBuffer = 0;
    glCreateBuffers(1, &faceBuffer);
    glNamedBufferStorage(faceBuffer, cloudFaces.size(), cloudFaces.data(), 0);

    GLuint faceTexture = 0;
    glCreateTextures(GL_TEXTURE_BUFFER, 1, &faceTexture);
    glTextureBuffer(faceTexture, GL_R8I, faceBuffer);
    glUniform1i(glGetUniformLocation(program, "CloudFaces"), 0);
    glBindTextureUnit(0, faceTexture);

    struct CloudInfo {
        float cloudColor[4];
        float cloudOffset[4];
        float cellSize[4];
    } cloudInfo = {
        {1.0f, 1.0f, 1.0f, 1.0f},
        {0.0f, 0.0f, 0.0f, 0.0f},
        {1.90f / 128.0f, 0.0f, 1.84f / 80.0f, 0.0f},
    };

    GLuint cloudUbo = 0;
    glCreateBuffers(1, &cloudUbo);
    glNamedBufferStorage(cloudUbo, sizeof(cloudInfo), &cloudInfo, 0);
    GLuint block = glGetUniformBlockIndex(program, "CloudInfo");
    glUniformBlockBinding(program, block, 3);
    glBindBufferRange(GL_UNIFORM_BUFFER, 3, cloudUbo, 0, sizeof(cloudInfo));

    GLuint vao = 0;
    glCreateVertexArrays(1, &vao);
    GLuint ebo = 0;
    glCreateBuffers(1, &ebo);
    glNamedBufferStorage(ebo, indices.size() * sizeof(indices[0]), indices.data(), 0);
    glVertexArrayElementBuffer(vao, ebo);

    GLuint color = makeTexture2D(kWidth, kHeight, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);
    GLuint depth = makeTexture2D(kWidth, kHeight, GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT);
    GLuint fbo = makeFbo(color, depth);

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glViewport(0, 0, kWidth, kHeight);
    glDisable(GL_SCISSOR_TEST);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glDepthMask(GL_TRUE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glClearColor(0.02f, 0.03f, 0.04f, 1.0f);
    glClearDepth(1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glBindVertexArray(vao);
    glDrawElements(GL_TRIANGLES, indexCount, GL_UNSIGNED_INT, nullptr);
    checkGl("cloud draw");

    auto pixels = readBack(kWidth, kHeight);
    writePpm("cloud-tbo-vertexid", kWidth, kHeight, pixels);

    bool ok = true;
    ok &= expectDominant("cloud sample A", sample(pixels, kWidth, 32, 32), 'r');
    ok &= expectDominant("cloud sample B", sample(pixels, kWidth, 128, 128), 'b');
    return ok ? 0 : 1;
}

static int runRtPingpongBlur()
{
    createMglWindow("rt-pingpong-blur");

    static const char *screenVs = R"GLSL(
#version 330
out vec2 texCoord;
void main() {
    vec2 uv = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    texCoord = uv;
    gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
)GLSL";

    static const char *seedFs = R"GLSL(
#version 330
in vec2 texCoord;
out vec4 fragColor;
void main() {
    vec2 p = texCoord;
    float stripe = step(0.5, fract((p.x + p.y) * 12.0));
    fragColor = vec4(p.x, p.y, 1.0 - p.x, 1.0);
    fragColor.rgb = mix(fragColor.rgb, vec3(1.0, 0.15, 0.05), stripe * 0.35);
}
)GLSL";

    static const char *blurFs = R"GLSL(
#version 330
uniform sampler2D InSampler;

layout(std140) uniform SamplerInfo {
    vec2 OutSize;
    vec2 InSize;
};

layout(std140) uniform BlurConfig {
    vec2 BlurDir;
    float Radius;
};

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec2 oneTexel = 1.0 / InSize;
    vec2 sampleStep = oneTexel * BlurDir;
    vec4 blurred = vec4(0.0);
    float actualRadius = round(Radius);
    for (float a = -actualRadius + 0.5; a <= actualRadius; a += 2.0) {
        blurred += texture(InSampler, texCoord + sampleStep * a);
    }
    blurred += texture(InSampler, texCoord + sampleStep * actualRadius) / 2.0;
    fragColor = blurred / (actualRadius + 0.5);
}
)GLSL";

    GLuint seedProgram = linkProgram(2, GL_VERTEX_SHADER, screenVs, GL_FRAGMENT_SHADER, seedFs);
    GLuint blurProgram = linkProgram(2, GL_VERTEX_SHADER, screenVs, GL_FRAGMENT_SHADER, blurFs);

    GLuint texA = makeTexture2D(kWidth, kHeight, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);
    GLuint texB = makeTexture2D(kWidth, kHeight, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);
    GLuint depthA = makeTexture2D(kWidth, kHeight, GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT);
    GLuint depthB = makeTexture2D(kWidth, kHeight, GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT);
    GLuint fboA = makeFbo(texA, depthA);
    GLuint fboB = makeFbo(texB, depthB);

    GLuint vao = 0;
    glCreateVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glBindFramebuffer(GL_FRAMEBUFFER, fboA);
    glViewport(0, 0, kWidth, kHeight);
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glDisable(GL_BLEND);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUseProgram(seedProgram);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    checkGl("seed draw");

    struct SamplerInfo {
        float outSize[2];
        float inSize[2];
    } samplerInfo = {{(float)kWidth, (float)kHeight}, {(float)kWidth, (float)kHeight}};
    GLuint samplerUbo = 0;
    glCreateBuffers(1, &samplerUbo);
    glNamedBufferStorage(samplerUbo, sizeof(samplerInfo), &samplerInfo, 0);

    struct BlurConfig {
        float blurDir[2];
        float radius;
        float pad;
    } blurConfig = {{1.0f, 0.0f}, 5.0f, 0.0f};
    GLuint blurUbo = 0;
    glCreateBuffers(1, &blurUbo);
    glNamedBufferStorage(blurUbo, sizeof(blurConfig), &blurConfig, GL_DYNAMIC_STORAGE_BIT);

    GLuint sampler = 0;
    glCreateSamplers(1, &sampler);
    glSamplerParameteri(sampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glSamplerParameteri(sampler, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glSamplerParameteri(sampler, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glSamplerParameteri(sampler, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glUseProgram(blurProgram);
    glUniform1i(glGetUniformLocation(blurProgram, "InSampler"), 0);
    glUniformBlockBinding(blurProgram, glGetUniformBlockIndex(blurProgram, "SamplerInfo"), 1);
    glUniformBlockBinding(blurProgram, glGetUniformBlockIndex(blurProgram, "BlurConfig"), 2);
    glBindBufferRange(GL_UNIFORM_BUFFER, 1, samplerUbo, 0, sizeof(samplerInfo));
    glBindBufferRange(GL_UNIFORM_BUFFER, 2, blurUbo, 0, 12);
    glBindSampler(0, sampler);

    GLuint srcTex = texA;
    GLuint dstFbo = fboB;
    for (int pass = 0; pass < 6; ++pass) {
        blurConfig.blurDir[0] = (pass & 1) ? 0.0f : 1.0f;
        blurConfig.blurDir[1] = (pass & 1) ? 1.0f : 0.0f;
        glNamedBufferSubData(blurUbo, 0, sizeof(blurConfig), &blurConfig);

        glBindFramebuffer(GL_FRAMEBUFFER, dstFbo);
        glViewport(0, 0, kWidth, kHeight);
        glUseProgram(blurProgram);
        glBindTextureUnit(0, srcTex);
        glBindSampler(0, sampler);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        checkGl("blur draw");

        if (srcTex == texA) {
            srcTex = texB;
            dstFbo = fboA;
        } else {
            srcTex = texA;
            dstFbo = fboB;
        }
    }

    GLuint finalFbo = (srcTex == texA) ? fboA : fboB;
    glBindFramebuffer(GL_READ_FRAMEBUFFER, finalFbo);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    auto pixels = readBack(kWidth, kHeight);
    writePpm("rt-pingpong-blur", kWidth, kHeight, pixels);

    Pixel bottomLeft = sample(pixels, kWidth, 20, 20);
    Pixel topRight = sample(pixels, kWidth, kWidth - 21, kHeight - 21);
    Pixel center = sample(pixels, kWidth, kWidth / 2, kHeight / 2);

    bool ok = true;
    ok &= bottomLeft.b > bottomLeft.r + 25;
    ok &= topRight.r > topRight.b + 25 && topRight.g > 120;
    ok &= center.r > 80 && center.g > 70 && center.b > 60;
    fprintf(stderr, "blur bottom-left rgba=(%u,%u,%u,%u)\n", bottomLeft.r, bottomLeft.g, bottomLeft.b, bottomLeft.a);
    fprintf(stderr, "blur top-right   rgba=(%u,%u,%u,%u)\n", topRight.r, topRight.g, topRight.b, topRight.a);
    fprintf(stderr, "blur center      rgba=(%u,%u,%u,%u)\n", center.r, center.g, center.b, center.a);
    fprintf(stderr, "rt-pingpong-blur => %s\n", ok ? "pass" : "FAIL");
    return ok ? 0 : 1;
}

static int runFocusedTailRendergraph()
{
    createMglWindow("focused-tail-rendergraph");

    static const char *skyVs = R"GLSL(
#version 330
layout(std140) uniform Projection {
    vec4 Tint;
    vec4 OffsetScale;
};
out vec3 cubeDir;
void main() {
    vec2 p = vec2((gl_VertexID == 1 || gl_VertexID == 2) ? 1.0 : -1.0,
                  (gl_VertexID >= 2) ? 1.0 : -1.0);
    cubeDir = vec3(p, 1.0);
    gl_Position = vec4(p * OffsetScale.zw + OffsetScale.xy, 0.6, 1.0);
}
)GLSL";

    static const char *skyFs = R"GLSL(
#version 330
uniform samplerCube Sampler0;
layout(std140) uniform DynamicTransforms {
    vec4 MulColor;
    vec4 AddColor;
};
in vec3 cubeDir;
out vec4 fragColor;
void main() {
    fragColor = texture(Sampler0, normalize(cubeDir)) * MulColor + AddColor;
}
)GLSL";

    static const char *guiVs = R"GLSL(
#version 330
layout(location = 0) in vec2 Position;
layout(location = 1) in vec2 TailUV0;
layout(location = 2) in vec2 ColorRG;
layout(std140) uniform Projection {
    vec4 Tint;
    vec4 OffsetScale;
};
layout(std140) uniform DynamicTransforms {
    vec4 MulColor;
    vec4 AddColor;
};
out vec2 texCoord;
out vec4 vertexColor;
void main() {
    texCoord = TailUV0;
    vertexColor = vec4(ColorRG, 1.0 - ColorRG.x * 0.35, 1.0) * MulColor + AddColor;
    gl_Position = vec4(Position * OffsetScale.zw + OffsetScale.xy, 0.0, 1.0);
}
)GLSL";

    static const char *guiFs = R"GLSL(
#version 330
uniform sampler2D Sampler0;
in vec2 texCoord;
in vec4 vertexColor;
out vec4 fragColor;
void main() {
    vec4 texel = texture(Sampler0, texCoord);
    fragColor = texel * vertexColor;
}
)GLSL";

    static const char *dualVs = R"GLSL(
#version 330
layout(location = 0) in vec2 Position;
layout(location = 1) in vec2 TailUV0;
layout(location = 2) in vec2 TailUV1;
layout(location = 3) in float MixFactor;
layout(std140) uniform Projection {
    vec4 Tint;
    vec4 OffsetScale;
};
layout(std140) uniform DynamicTransforms {
    vec4 MulColor;
    vec4 AddColor;
};
layout(std140) uniform ExtraInfo {
    vec4 Params0;
    vec4 Params1;
};
out vec2 texCoord0;
out vec2 texCoord1;
out float mixFactor;
out vec4 vertexColor;
void main() {
    texCoord0 = TailUV0;
    texCoord1 = TailUV1;
    mixFactor = MixFactor;
    vertexColor = MulColor + AddColor + Params0 * 0.10;
    gl_Position = vec4(Position * OffsetScale.zw + OffsetScale.xy, 0.0, 1.0);
}
)GLSL";

    static const char *dualFs = R"GLSL(
#version 330
uniform sampler2D Sampler0;
uniform sampler2D Sampler1;
in vec2 texCoord0;
in vec2 texCoord1;
in float mixFactor;
in vec4 vertexColor;
out vec4 fragColor;
void main() {
    vec4 a = texture(Sampler0, texCoord0);
    vec4 b = texture(Sampler1, texCoord1);
    fragColor = mix(a, b, mixFactor) * vertexColor;
}
)GLSL";

    static const char *screenVs = R"GLSL(
#version 330
out vec2 texCoord;
void main() {
    vec2 uv = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    texCoord = uv;
    gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
)GLSL";

    static const char *blurFs = R"GLSL(
#version 330
uniform sampler2D InSampler;
layout(std140) uniform SamplerInfo {
    vec2 OutSize;
    vec2 InSize;
};
layout(std140) uniform BlurConfig {
    vec2 BlurDir;
    float Radius;
};
layout(std140) uniform TailGlobals {
    vec4 ColorModulator;
    vec4 PassInfo;
    vec4 Spare0;
    vec2 Spare1;
};
in vec2 texCoord;
out vec4 fragColor;
void main() {
    vec2 oneTexel = 1.0 / InSize;
    vec2 sampleStep = oneTexel * BlurDir;
    vec4 sum = texture(InSampler, texCoord) * 0.32;
    sum += texture(InSampler, texCoord + sampleStep * 1.5) * 0.23;
    sum += texture(InSampler, texCoord - sampleStep * 1.5) * 0.23;
    sum += texture(InSampler, texCoord + sampleStep * Radius) * 0.11;
    sum += texture(InSampler, texCoord - sampleStep * Radius) * 0.11;
    fragColor = sum * ColorModulator + vec4(PassInfo.x * 0.015, PassInfo.y * 0.010, 0.0, 0.0);
}
)GLSL";

    GLuint skyProgram = linkProgram(2, GL_VERTEX_SHADER, skyVs, GL_FRAGMENT_SHADER, skyFs);
    GLuint guiProgram = linkProgram(2, GL_VERTEX_SHADER, guiVs, GL_FRAGMENT_SHADER, guiFs);
    GLuint dualProgram = linkProgram(2, GL_VERTEX_SHADER, dualVs, GL_FRAGMENT_SHADER, dualFs);
    GLuint blurPrograms[6] = {};
    for (int i = 0; i < 6; ++i) {
        blurPrograms[i] = linkProgram(2, GL_VERTEX_SHADER, screenVs, GL_FRAGMENT_SHADER, blurFs);
    }

    GLuint colorScene = makeTexture2D(kWidth, kHeight, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);
    GLuint depthScene = makeTexture2D(kWidth, kHeight, GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT);
    GLuint colorScratch = makeTexture2D(kWidth, kHeight, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);
    GLuint depthScratch = makeTexture2D(kWidth, kHeight, GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT);
    GLuint defaultColor = makeTexture2D(kWidth, kHeight, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);

    GLuint fboScene = makeFbo(colorScene, depthScene);
    GLuint fboScratch = makeFbo(colorScratch, depthScratch);
    GLuint fboDefaultProxy = makeFbo(defaultColor, 0);

    GLuint samplerLinear = makeSampler(GL_LINEAR, GL_LINEAR);
    GLuint samplerNearest = makeSampler(GL_NEAREST, GL_NEAREST);
    GLuint samplerBlur = makeSampler(GL_LINEAR, GL_LINEAR);

    GLuint cubeTex = makeCubeTexture();
    GLuint texGuiA = makeCheckerTexture(kWidth, kHeight, 220, 60, 40, 60, 140, 240);
    GLuint texGuiB = makeCheckerTexture(kWidth, kHeight, 45, 220, 120, 235, 210, 40);
    GLuint texDual0 = makeCheckerTexture(kWidth, kHeight, 170, 40, 220, 40, 190, 230);
    GLuint texDual1 = makeCheckerTexture(kWidth, kHeight, 240, 130, 35, 35, 75, 210);

    GLuint projectionUbo = makePersistentBuffer(512);
    GLuint dynamicUboA = makePersistentBuffer(2048);
    GLuint dynamicUboB = makePersistentBuffer(2048);
    GLuint extraUbo = 0;
    glCreateBuffers(1, &extraUbo);
    float extraInfo[12] = {0.85f, 0.95f, 1.10f, 1.0f, 0.15f, 0.25f, 0.35f, 0.45f,
                           0.05f, 0.10f, 0.15f, 0.20f};
    glNamedBufferStorage(extraUbo, sizeof(extraInfo), extraInfo, 0);

    float projection0[16] = {1.0f, 0.92f, 0.88f, 1.0f, 0.0f, 0.0f, 0.96f, 0.96f};
    float projection1[16] = {0.95f, 1.0f, 0.96f, 1.0f, 0.01f, -0.02f, 0.92f, 0.94f};
    writeMappedRange(projectionUbo, 0, 256, projection0);
    writeMappedRange(projectionUbo, 256, 256, projection1);

    float dyn0[16] = {0.95f, 0.90f, 0.85f, 1.0f, 0.03f, 0.02f, 0.01f, 0.0f};
    float dyn1[16] = {0.70f, 0.95f, 1.15f, 1.0f, 0.05f, 0.01f, 0.04f, 0.0f};
    writeMappedRange(dynamicUboA, 0, 96, dyn0);
    writeMappedRange(dynamicUboA, 96, 1440, dyn0);
    writeMappedRange(dynamicUboA, 1536, 96, dyn1);
    writeMappedRange(dynamicUboB, 0, 96, dyn1);
    writeMappedRange(dynamicUboB, 96, 96, dyn0);
    writeMappedRange(dynamicUboB, 256, 256, dyn1);

    std::vector<TailVertex> vertices(84);
    fillTailVertices(vertices);
    GLuint guiVbo = makePersistentBuffer((GLsizeiptr)(vertices.size() * sizeof(vertices[0])));
    writeMappedRange(guiVbo, 0, (GLsizeiptr)(vertices.size() * sizeof(vertices[0])), vertices.data());

    std::vector<TailDualVertex> dualVertices(1240);
    fillTailDualVertices(dualVertices);
    GLuint dualVbo = makePersistentBuffer((GLsizeiptr)(dualVertices.size() * sizeof(dualVertices[0])));
    writeMappedRange(dualVbo, 0, 11200, dualVertices.data());
    writeMappedRange(dualVbo, 11200, 15680,
                     (const uint8_t *)dualVertices.data() + 11200);

    std::vector<uint32_t> indices(900);
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = (uint32_t)(i % 84);
    }
    GLuint ebo = 0;
    glCreateBuffers(1, &ebo);
    glNamedBufferStorage(ebo, indices.size() * sizeof(indices[0]), indices.data(), 0);

    GLuint guiVao = makeTailVao(guiVbo, ebo);
    GLuint dualVao = makeTailDualVao(dualVbo, ebo);
    GLuint screenVao = makeFullscreenVao();

    struct BlurInfo {
        float outSize[2];
        float inSize[2];
    };
    struct BlurConfig {
        float blurDir[2];
        float radius;
        float pad;
    };
    struct TailGlobals {
        float colorModulator[4];
        float passInfo[4];
        float spare0[4];
        float spare1[2];
    };

    TailGlobals globals = {
        {1.0f, 0.96f, 0.92f, 1.0f},
        {0.0f, 0.0f, 0.0f, 0.0f},
        {0.1f, 0.2f, 0.3f, 0.4f},
        {0.5f, 0.6f},
    };
    GLuint tailGlobalsUbo = 0;
    glCreateBuffers(1, &tailGlobalsUbo);
    glNamedBufferStorage(tailGlobalsUbo, sizeof(globals), &globals, GL_DYNAMIC_STORAGE_BIT);

    GLuint blurInfoUbos[6] = {};
    GLuint blurConfigUbos[6] = {};
    for (int i = 0; i < 6; ++i) {
        BlurInfo info = {{(float)kWidth, (float)kHeight}, {(float)kWidth, (float)kHeight}};
        BlurConfig cfg = {{(i & 1) ? 0.0f : 1.0f, (i & 1) ? 1.0f : 0.0f}, 5.0f, 0.0f};
        glCreateBuffers(1, &blurInfoUbos[i]);
        glNamedBufferStorage(blurInfoUbos[i], sizeof(info), &info, GL_MAP_WRITE_BIT | GL_DYNAMIC_STORAGE_BIT);
        glNamedBufferSubData(blurInfoUbos[i], 0, sizeof(info), &info);
        glCreateBuffers(1, &blurConfigUbos[i]);
        glNamedBufferStorage(blurConfigUbos[i], sizeof(cfg), &cfg, GL_MAP_WRITE_BIT | GL_DYNAMIC_STORAGE_BIT);
        glNamedBufferSubData(blurConfigUbos[i], 0, sizeof(cfg), &cfg);
    }

    auto bindProgramBlocks = [](GLuint program) {
        GLuint projection = glGetUniformBlockIndex(program, "Projection");
        if (projection != GL_INVALID_INDEX) glUniformBlockBinding(program, projection, 1);
        GLuint dynamic = glGetUniformBlockIndex(program, "DynamicTransforms");
        if (dynamic != GL_INVALID_INDEX) glUniformBlockBinding(program, dynamic, 0);
        GLuint extra = glGetUniformBlockIndex(program, "ExtraInfo");
        if (extra != GL_INVALID_INDEX) glUniformBlockBinding(program, extra, 2);
        GLuint samplerInfo = glGetUniformBlockIndex(program, "SamplerInfo");
        if (samplerInfo != GL_INVALID_INDEX) glUniformBlockBinding(program, samplerInfo, 0);
        GLuint blurConfig = glGetUniformBlockIndex(program, "BlurConfig");
        if (blurConfig != GL_INVALID_INDEX) glUniformBlockBinding(program, blurConfig, 1);
        GLuint tailGlobals = glGetUniformBlockIndex(program, "TailGlobals");
        if (tailGlobals != GL_INVALID_INDEX) glUniformBlockBinding(program, tailGlobals, 2);
    };
    bindProgramBlocks(skyProgram);
    bindProgramBlocks(guiProgram);
    bindProgramBlocks(dualProgram);
    for (GLuint p : blurPrograms) {
        bindProgramBlocks(p);
        glUseProgram(p);
        glUniform1i(glGetUniformLocation(p, "InSampler"), 0);
    }
    glUseProgram(guiProgram);
    glUniform1i(glGetUniformLocation(guiProgram, "Sampler0"), 0);
    glUseProgram(dualProgram);
    glUniform1i(glGetUniformLocation(dualProgram, "Sampler0"), 0);
    glUniform1i(glGetUniformLocation(dualProgram, "Sampler1"), 1);
    glUseProgram(skyProgram);
    glUniform1i(glGetUniformLocation(skyProgram, "Sampler0"), 0);

    glBindFramebuffer(GL_FRAMEBUFFER, fboScratch);
    glClearDepth(1.0);
    glClearColor(0, 0, 0, 0);
    glDepthMask(GL_TRUE);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    waitOnFence();

    glNamedFramebufferTexture(fboDefaultProxy, GL_COLOR_ATTACHMENT0, 0, 0);
    glNamedFramebufferTexture(fboDefaultProxy, GL_DEPTH_ATTACHMENT, depthScene, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, fboDefaultProxy);
    glDrawBuffer(GL_ZERO);
    glClearDepth(1.0);
    glClear(GL_DEPTH_BUFFER_BIT);
    glDrawBuffer(GL_COLOR_ATTACHMENT0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0);
    waitOnFence();

    glBindFramebuffer(GL_FRAMEBUFFER, fboScene);
    glViewport(0, 0, kWidth, kHeight);
    glEnable(GL_DEPTH_TEST);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glDepthMask(GL_FALSE);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_FALSE);
    glUseProgram(skyProgram);
    glBindTextureUnit(0, cubeTex);
    glBindSampler(0, samplerLinear);
    glBindBufferRange(GL_UNIFORM_BUFFER, 1, projectionUbo, 0, 64);
    glBindBufferRange(GL_UNIFORM_BUFFER, 0, dynamicUboA, 0, 256);
    glBindVertexArray(screenVao);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    checkGl("tail sky draw");
    waitOnFence();

    glBindFramebuffer(GL_FRAMEBUFFER, fboScene);
    glViewport(0, 0, kWidth, kHeight);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glDepthMask(GL_TRUE);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glUseProgram(guiProgram);
    glBindTextureUnit(0, texGuiA);
    glBindSampler(0, samplerNearest);
    glBindBufferRange(GL_UNIFORM_BUFFER, 1, projectionUbo, 0, 64);
    glBindBufferRange(GL_UNIFORM_BUFFER, 0, dynamicUboA, 256, 256);
    glBindVertexArray(guiVao);
    glBindVertexBuffer(0, guiVbo, 0, sizeof(TailVertex));
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
    glBindTextureUnit(0, texGuiB);
    glBindSampler(0, samplerLinear);
    glDrawElementsBaseVertex(GL_TRIANGLES, 90, GL_UNSIGNED_INT, nullptr, 4);
    glBindTextureUnit(0, texGuiA);
    glBindSampler(0, samplerNearest);
    glDrawElementsBaseVertex(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr, 64);
    checkGl("tail gui program 81 draws");

    glUseProgram(dualProgram);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, texDual1);
    glBindSampler(1, samplerBlur);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texDual0);
    glBindSampler(0, samplerLinear);
    glBindBufferRange(GL_UNIFORM_BUFFER, 1, projectionUbo, 0, 64);
    glBindBufferRange(GL_UNIFORM_BUFFER, 0, dynamicUboA, 256, 256);
    glBindBufferRange(GL_UNIFORM_BUFFER, 2, extraUbo, 0, 40);
    glBindVertexArray(dualVao);
    glBindVertexBuffer(0, dualVbo, 0, sizeof(TailDualVertex));
    glDrawElements(GL_TRIANGLES, 600, GL_UNSIGNED_INT, nullptr);
    glUseProgram(guiProgram);
    glBindTextureUnit(0, texGuiB);
    glBindSampler(0, samplerLinear);
    glBindVertexArray(guiVao);
    glDrawElementsBaseVertex(GL_TRIANGLES, 12, GL_UNSIGNED_INT, nullptr, 68);
    glBindTextureUnit(0, texGuiA);
    glBindSampler(0, samplerNearest);
    glDrawElementsBaseVertex(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr, 76);
    glUseProgram(dualProgram);
    glBindTextureUnit(0, texDual0);
    glBindSampler(0, samplerLinear);
    glBindBufferRange(GL_UNIFORM_BUFFER, 2, extraUbo, 0, 40);
    glBindVertexArray(dualVao);
    glDrawElementsBaseVertex(GL_TRIANGLES, 840, GL_UNSIGNED_INT, nullptr, 400);
    checkGl("tail dual/gui mixed draws");

    glBindFramebuffer(GL_READ_FRAMEBUFFER, fboScene);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fboDefaultProxy);
    glViewport(0, 0, kWidth, kHeight);
    glNamedFramebufferTexture(fboDefaultProxy, GL_COLOR_ATTACHMENT0, defaultColor, 0);
    glNamedFramebufferTexture(fboDefaultProxy, GL_DEPTH_ATTACHMENT, 0, 0);
    glBlitNamedFramebuffer(fboScene, fboDefaultProxy, 0, 0, kWidth, kHeight, 0, 0, kWidth, kHeight,
                           GL_COLOR_BUFFER_BIT, GL_NEAREST);
    checkGl("tail first blit");
    waitOnFence();

    glBindFramebuffer(GL_FRAMEBUFFER, fboScratch);
    glClearDepth(1.0);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glBindFramebuffer(GL_FRAMEBUFFER, fboDefaultProxy);
    glDrawBuffer(GL_ZERO);
    glClearDepth(1.0);
    glDepthMask(GL_TRUE);
    glClear(GL_DEPTH_BUFFER_BIT);
    glDrawBuffer(GL_COLOR_ATTACHMENT0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0);

    GLuint runtimeDepth = makeTexture2D(kWidth, kHeight, GL_DEPTH_COMPONENT32, GL_DEPTH_COMPONENT, GL_FLOAT);
    GLuint runtimeColor = makeTexture2D(kWidth, kHeight, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);
    GLuint fboRuntimeA = 0;
    GLuint fboRuntimeB = 0;
    glCreateFramebuffers(1, &fboRuntimeA);
    glNamedFramebufferTexture(fboRuntimeA, GL_COLOR_ATTACHMENT0, runtimeColor, 0);
    glNamedFramebufferTexture(fboRuntimeA, GL_DEPTH_ATTACHMENT, runtimeDepth, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, fboRuntimeA);
    glClearDepth(1.0);
    glClearColor(0, 0, 0, 0);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    waitOnFence();

    glCreateFramebuffers(1, &fboRuntimeB);
    glNamedFramebufferTexture(fboRuntimeB, GL_COLOR_ATTACHMENT0, runtimeColor, 0);
    glNamedFramebufferTexture(fboRuntimeB, GL_DEPTH_ATTACHMENT, runtimeDepth, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, fboRuntimeB);
    glViewport(0, 0, kWidth, kHeight);
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);

    GLuint dstFbos[6] = {fboRuntimeB, fboScene, fboRuntimeB, fboScene, fboRuntimeB, fboScene};
    GLuint srcTexs[6] = {colorScene, runtimeColor, colorScene, runtimeColor, colorScene, runtimeColor};
    for (int pass = 0; pass < 6; ++pass) {
        globals.passInfo[0] = (float)pass;
        globals.passInfo[1] = (pass & 1) ? 1.0f : 0.0f;
        glNamedBufferSubData(tailGlobalsUbo, 0, sizeof(globals), &globals);
        glBindFramebuffer(GL_FRAMEBUFFER, dstFbos[pass]);
        glViewport(0, 0, kWidth, kHeight);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glUseProgram(blurPrograms[pass]);
        glBindBufferRange(GL_UNIFORM_BUFFER, 2, tailGlobalsUbo, 0, 56);
        glBindBufferRange(GL_UNIFORM_BUFFER, 0, blurInfoUbos[pass], 0, 16);
        glUniform1i(glGetUniformLocation(blurPrograms[pass], "InSampler"), 0);
        glBindTextureUnit(0, srcTexs[pass]);
        glBindSampler(0, samplerBlur);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
        glBindBufferRange(GL_UNIFORM_BUFFER, 1, blurConfigUbos[pass], 0, 12);
        glBindVertexArray(screenVao);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        checkGl("tail blur draw");
        waitOnFence();
    }

    glBindFramebuffer(GL_FRAMEBUFFER, fboScene);
    glViewport(0, 0, kWidth, kHeight);
    glEnable(GL_BLEND);
    glDepthMask(GL_TRUE);
    glUseProgram(guiProgram);
    glBindTextureUnit(0, texGuiA);
    glBindSampler(0, samplerNearest);
    glBindBufferRange(GL_UNIFORM_BUFFER, 1, projectionUbo, 0, 64);
    glBindBufferRange(GL_UNIFORM_BUFFER, 0, dynamicUboB, 256, 256);
    glBindVertexArray(guiVao);
    glBindVertexBuffer(0, guiVbo, 0, sizeof(TailVertex));
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
    glBindTextureUnit(0, texGuiB);
    glDrawElementsBaseVertex(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr, 4);
    checkGl("tail post-blur gui draws");

    glBindFramebuffer(GL_READ_FRAMEBUFFER, fboScene);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fboDefaultProxy);
    glViewport(0, 0, kWidth, kHeight);
    glNamedFramebufferTexture(fboDefaultProxy, GL_COLOR_ATTACHMENT0, defaultColor, 0);
    glNamedFramebufferTexture(fboDefaultProxy, GL_DEPTH_ATTACHMENT, 0, 0);
    glBlitNamedFramebuffer(fboScene, fboDefaultProxy, 0, 0, kWidth, kHeight, 0, 0, kWidth, kHeight,
                           GL_COLOR_BUFFER_BIT, GL_NEAREST);
    checkGl("tail final blit");

    glBindFramebuffer(GL_READ_FRAMEBUFFER, fboDefaultProxy);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    auto pixels = readBack(kWidth, kHeight);
    writePpm("focused-tail-rendergraph", kWidth, kHeight, pixels);

    Pixel bottomLeft = sample(pixels, kWidth, 24, 24);
    Pixel center = sample(pixels, kWidth, kWidth / 2, kHeight / 2);
    Pixel topRight = sample(pixels, kWidth, kWidth - 25, kHeight - 25);
    bool ok = true;
    ok &= (bottomLeft.r + bottomLeft.g + bottomLeft.b) > 80;
    ok &= (center.r + center.g + center.b) > 120;
    ok &= (topRight.r + topRight.g + topRight.b) > 80;
    fprintf(stderr, "tail bottom-left rgba=(%u,%u,%u,%u)\n", bottomLeft.r, bottomLeft.g, bottomLeft.b, bottomLeft.a);
    fprintf(stderr, "tail center      rgba=(%u,%u,%u,%u)\n", center.r, center.g, center.b, center.a);
    fprintf(stderr, "tail top-right   rgba=(%u,%u,%u,%u)\n", topRight.r, topRight.g, topRight.b, topRight.a);
    fprintf(stderr, "focused-tail-rendergraph => %s\n", ok ? "pass" : "FAIL");
    return ok ? 0 : 1;
}

int main(int argc, char **argv)
{
    @autoreleasepool {
        const char *which = argc > 1 ? argv[1] : "all";
        int result = 0;

        if (strcmp(which, "cloud-tbo-vertexid") == 0 || strcmp(which, "cloud") == 0 || strcmp(which, "all") == 0) {
            result |= runCloudTboVertexId();
            glfwTerminate();
        }

        if (strcmp(which, "rt-pingpong-blur") == 0 || strcmp(which, "blur") == 0 || strcmp(which, "all") == 0) {
            result |= runRtPingpongBlur();
            glfwTerminate();
        }

        if (strcmp(which, "focused-tail-rendergraph") == 0 || strcmp(which, "tail") == 0 || strcmp(which, "all") == 0) {
            result |= runFocusedTailRendergraph();
            glfwTerminate();
        }

        if (strcmp(which, "cloud-tbo-vertexid") != 0 && strcmp(which, "cloud") != 0 &&
            strcmp(which, "rt-pingpong-blur") != 0 && strcmp(which, "blur") != 0 &&
            strcmp(which, "focused-tail-rendergraph") != 0 && strcmp(which, "tail") != 0 &&
            strcmp(which, "all") != 0) {
            fprintf(stderr, "usage: %s [all|cloud-tbo-vertexid|rt-pingpong-blur|focused-tail-rendergraph]\n", argv[0]);
            return 2;
        }

        return result;
    }
}
