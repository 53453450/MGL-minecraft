/*
 * Michael Larson on 1/6/2022
 *
 * mgl_context.m
 * GLFW
 *
 */

#import <QuartzCore/QuartzCore.h>

#include "MGLContext.h"
#include "internal.h"
#include "MGLRenderer.h"

#include <unistd.h>
#include <math.h>
#include <dlfcn.h>
#include <limits.h>
#include <string.h>

#define GL_BGRA                           0x80E1
#define GL_UNSIGNED_INT_8_8_8_8_REV       0x8367
#define GL_DEPTH_COMPONENT                0x1902
#define GL_FLOAT                          0x1406


GLMContext createGLMContext(GLenum format, GLenum type,
                        GLenum depth_format, GLenum depth_type,
                        GLenum stencil_format, GLenum stencil_type);

void MGLsetDefaultFramebufferSRGBCapable(GLMContext ctx, GLboolean capable);

void MGLsetCurrentContext(GLMContext ctx);
GLMContext MGLgetCurrentContext(void);
void MGLswapBuffers(GLMContext ctx);
void destroyGLMContext(GLMContext ctx);

static void makeContextCurrentMGL(_GLFWwindow* window)
{
    @autoreleasepool {

    if (window)
    {
        MGLsetCurrentContext(window->context.mgl.ctx);

        _glfwPlatformSetTls(&_glfw.contextSlot, window);
    }
    else
    {
        MGLsetCurrentContext(NULL);
        _glfwPlatformSetTls(&_glfw.contextSlot, NULL);
    }

    } // autoreleasepool
}

static void swapBuffersMGL(_GLFWwindow* window)
{
    MGLswapBuffers(window->context.mgl.ctx);
}

static void swapIntervalMGL(int interval)
{

}

static int extensionSupportedMGL(const char* extension)
{
    if (!extension) {
        return GLFW_FALSE;
    }

    static const char* supportedExtensions[] = {
        "GL_ARB_vertex_array_object",
        "GL_ARB_framebuffer_object",
        "GL_ARB_texture_storage",
        "GL_ARB_sampler_objects",
        "GL_ARB_uniform_buffer_object",
        "GL_ARB_draw_buffers",
        "GL_ARB_debug_output",
        "GL_ARB_texture_buffer_object",
        "GL_ARB_texture_buffer_range",
        "GL_ARB_buffer_storage",
        "GL_ARB_direct_state_access"
    };

    for (size_t i = 0; i < sizeof(supportedExtensions) / sizeof(supportedExtensions[0]); i++) {
        if (strcmp(extension, supportedExtensions[i]) == 0) {
            return GLFW_TRUE;
        }
    }

    return GLFW_FALSE;
}

static GLFWglproc getProcAddressMGL(const char* procname)
{
    GLFWproc symbol;
    if (!_glfw.mgl.handle)
        return NULL;

    symbol = _glfwPlatformGetModuleSymbol(_glfw.mgl.handle, procname);
    return symbol;
}

static void destroyContextMGL(_GLFWwindow* window)
{
    @autoreleasepool {
        if (!window)
            return;

        if (window->context.mgl.ctx)
        {
            destroyGLMContext(window->context.mgl.ctx);
            window->context.mgl.ctx = NULL;
        }

        window->context.mgl.renderer = nil;

    } // autoreleasepool
}


//////////////////////////////////////////////////////////////////////////
//////                       GLFW internal API                      //////
//////////////////////////////////////////////////////////////////////////

// Initialize OpenGL support
//
GLFWbool _glfwInitMGL(void)
{
    Dl_info info;
    char modulePath[PATH_MAX];
    const char* slash;

    if (_glfw.mgl.handle)
        return GLFW_TRUE;

    // Fast path: rely on platform loader search rules first.
    _glfw.mgl.handle = _glfwPlatformLoadModule("libmgl.dylib");

    // Robust fallback for Java/LWJGL launchers: load libmgl from the same
    // directory as the currently loaded libglfw.dylib.
    if (_glfw.mgl.handle == NULL &&
        dladdr((const void*) _glfwInitMGL, &info) != 0 &&
        info.dli_fname != NULL)
    {
        slash = strrchr(info.dli_fname, '/');
        if (slash)
        {
            size_t dirLen = (size_t) (slash - info.dli_fname);
            if (dirLen + 1 + strlen("libmgl.dylib") + 1 < sizeof(modulePath))
            {
                memcpy(modulePath, info.dli_fname, dirLen);
                modulePath[dirLen] = '/';
                strcpy(modulePath + dirLen + 1, "libmgl.dylib");
                modulePath[dirLen + 1 + strlen("libmgl.dylib")] = '\0';
                _glfw.mgl.handle = _glfwPlatformLoadModule(modulePath);
            }
        }
    }

    if (_glfw.mgl.handle == NULL)
    {
        _glfwInputError(GLFW_API_UNAVAILABLE,
                        "MGL: Failed to locate libmgl.dylib");
        return GLFW_FALSE;
    }

    return GLFW_TRUE;
}

// Terminate OpenGL support
//
void _glfwTerminateMGL(void)
{
}

// Create the OpenGL context
//
GLFWbool _glfwCreateContextMGL(_GLFWwindow* window,
                                const _GLFWctxconfig* ctxconfig,
                                const _GLFWfbconfig* fbconfig)
{
    if (ctxconfig->client == GLFW_OPENGL_ES_API)
    {
        _glfwInputError(GLFW_API_UNAVAILABLE,
                        "MGL: OpenGL ES is not available on macOS");
        return GLFW_FALSE;
    }

    // MGL internally targets a modern core feature set, but the OpenGL CTS
    // covers GL 3.0/3.1 packages before moving to 3.2+ core profile contexts.
    if (ctxconfig->major < 3 ||
        (ctxconfig->major == 3 && ctxconfig->minor < 0))
    {
        _glfwInputError(GLFW_VERSION_UNAVAILABLE,
                        "MGL: OpenGL 3.0+ required");
        return GLFW_FALSE;
    }

    window->context.mgl.ctx = createGLMContext(GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV,
                                               GL_DEPTH_COMPONENT, GL_FLOAT,
                                               0, 0);
    assert(window->context.mgl.ctx);
    
    // Apply GLFW_SRGB_CAPABLE hint to the default framebuffer.
    // When enabled, the Metal drawable will use _sRGB pixel format so that
    // fragment shader outputs are automatically encoded to sRGB on write.
    if (fbconfig && fbconfig->sRGB) {
        MGLsetDefaultFramebufferSRGBCapable(window->context.mgl.ctx, GLFW_TRUE);
    }
    
    if (window->context.mgl.ctx == nil)
    {
        _glfwInputError(GLFW_VERSION_UNAVAILABLE,
                        "MGL: Failed to create MGL context");
        return GLFW_FALSE;
    }

    [window->ns.view wantsLayer];

    MGLRenderer *renderer = [[MGLRenderer alloc] init];
    assert(renderer);

    window->context.mgl.renderer = (id)CFBridgingRetain(renderer);

    [window->context.mgl.renderer createMGLRendererAndBindToContext: window->context.mgl.ctx view: window->ns.view];

    //[window->context.mgl.object setView: window->ns.view];

    window->context.makeCurrent = makeContextCurrentMGL;
    window->context.swapBuffers = swapBuffersMGL;
    window->context.swapInterval = swapIntervalMGL;
    window->context.extensionSupported = extensionSupportedMGL;
    window->context.getProcAddress = getProcAddressMGL;
    window->context.destroy = destroyContextMGL;

    // Keep behavior robust for callers that create capabilities immediately
    // after window creation.
    makeContextCurrentMGL(window);

    return GLFW_TRUE;
}


//////////////////////////////////////////////////////////////////////////
//////                        GLFW native API                       //////
//////////////////////////////////////////////////////////////////////////

GLFWAPI void * glfwGetMGLContext(GLFWwindow* handle)
{
    _GLFWwindow* window = (_GLFWwindow*) handle;
    _GLFW_REQUIRE_INIT_OR_RETURN(nil);

    if (_glfw.platform.platformID != GLFW_PLATFORM_COCOA)
    {
        _glfwInputError(GLFW_PLATFORM_UNAVAILABLE,
                        "MGL: Platform not initialized");
        return nil;
    }

    if (window->context.source != GLFW_NATIVE_CONTEXT_API)
    {
        _glfwInputError(GLFW_NO_WINDOW_CONTEXT, NULL);
        return nil;
    }

    return window->context.mgl.ctx;
}
