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
#include "mgl_context_host_ops.h"
#include "mgl_glfw_abi.h"

#include <unistd.h>
#include <math.h>
#include <dlfcn.h>
#include <limits.h>
#include <string.h>

#define GL_BGRA                           0x80E1
#define GL_UNSIGNED_INT_8_8_8_8_REV       0x8367
#define GL_DEPTH_COMPONENT                0x1902
#define GL_FLOAT                          0x1406

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
    /* H2: plain C call through the ops table - no objc_msgSend to MGL
     * selectors is left in this TU, so a removed MGL entry can no longer
     * surface as an unrecognized-selector crash (mglSetSwapInterval:,
     * 2026-09-13). */
    _GLFWwindow* window = _glfwPlatformGetTls(&_glfw.contextSlot);
    if (window && window->context.mgl.ops && window->context.mgl.renderer) {
        window->context.mgl.ops->set_swap_interval(window->context.mgl.renderer,
                                                   interval);
    }
}

static int extensionSupportedMGL(const char* extension)
{
    if (!extension || !_glfw.mgl.handle) {
        return GLFW_FALSE;
    }

    typedef const unsigned char* (*PFNGLGETSTRINGIPROC)(unsigned int, unsigned int);
    typedef void (*PFNGLGETINTEGERVPROC)(unsigned int, int*);
    PFNGLGETSTRINGIPROC getStringi =
        (PFNGLGETSTRINGIPROC)_glfwPlatformGetModuleSymbol(_glfw.mgl.handle, "glGetStringi");
    PFNGLGETINTEGERVPROC getIntegerv =
        (PFNGLGETINTEGERVPROC)_glfwPlatformGetModuleSymbol(_glfw.mgl.handle, "glGetIntegerv");
    if (!getStringi || !getIntegerv) {
        return GLFW_FALSE;
    }

    int n = 0;
    getIntegerv(0x821D /* GL_NUM_EXTENSIONS */, &n);
    for (int i = 0; i < n; i++) {
        const unsigned char* ext = getStringi(0x1F03 /* GL_EXTENSIONS */, (unsigned int)i);
        if (ext && strcmp((const char*)ext, extension) == 0) {
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

        /* The renderer handle came from ops->create_and_bind holding one
         * reference; hand it back through the same ABI. */
        if (window->context.mgl.renderer) {
            if (window->context.mgl.ops)
                window->context.mgl.ops->release_owner(window->context.mgl.renderer);
            window->context.mgl.renderer = nil;
        }

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

    if (ctxconfig->share)
    {
        _glfwInputError(GLFW_INVALID_VALUE,
                        "MGL: shared GL contexts are not supported");
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
    if (!window->context.mgl.ctx)
    {
        _glfwInputError(GLFW_VERSION_UNAVAILABLE,
                        "MGL: Failed to allocate MGL context");
        return GLFW_FALSE;
    }
    
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

    /* H2: fetch the MGL-facing C ABI once and drive the renderer through it.
     * The renderer object is allocated and bound inside libmgl
     * (ops->create_and_bind); this TU never sends an MGL selector and never
     * references the MGLRenderer class, so a removed or renamed MGL entry is
     * a compile-time event for rebuilt consumers and a NULL check for stale
     * ones - not an unrecognized-selector crash. */
    const MGLContextHostOps* ops = mglContextHostOps();
    if (!ops || ops->version != MGL_CONTEXT_HOST_OPS_VERSION ||
        ops->size < sizeof(MGLContextHostOps))
    {
        destroyGLMContext(window->context.mgl.ctx);
        window->context.mgl.ctx = NULL;
        _glfwInputError(GLFW_VERSION_UNAVAILABLE,
                        "MGL: context host ops missing or incompatible");
        return GLFW_FALSE;
    }
    window->context.mgl.ops = ops;

    window->context.mgl.renderer = ops->create_and_bind(window->context.mgl.ctx,
                                                        window->ns.view);
    if (!window->context.mgl.renderer)
    {
        destroyGLMContext(window->context.mgl.ctx);
        window->context.mgl.ctx = NULL;
        _glfwInputError(GLFW_VERSION_UNAVAILABLE,
                        "MGL: Failed to allocate renderer");
        return GLFW_FALSE;
    }

    if (!ops->renderer_is_ready(window->context.mgl.renderer))
    {
        /* The renderer owns the backend and platform shell through the
         * context.  Destroy it before exposing any GLFW callbacks so a
         * failed Metal device/queue/layer setup cannot become a half-live
         * context. */
        destroyGLMContext(window->context.mgl.ctx);
        window->context.mgl.ctx = NULL;
        ops->release_owner(window->context.mgl.renderer);
        window->context.mgl.renderer = nil;
        _glfwInputError(GLFW_VERSION_UNAVAILABLE,
                        "MGL: Failed to initialize Metal renderer");
        return GLFW_FALSE;
    }

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
