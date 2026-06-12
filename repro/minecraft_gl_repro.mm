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

        if (strcmp(which, "cloud-tbo-vertexid") != 0 && strcmp(which, "cloud") != 0 &&
            strcmp(which, "rt-pingpong-blur") != 0 && strcmp(which, "blur") != 0 &&
            strcmp(which, "all") != 0) {
            fprintf(stderr, "usage: %s [all|cloud-tbo-vertexid|rt-pingpong-blur]\n", argv[0]);
            return 2;
        }

        return result;
    }
}
