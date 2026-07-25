/*
 * mgl_trace_log.m
 * MGL
 *
 * Implementation of the Trace Log Subsystem core infrastructure.
 * See mgl_trace_log.h for the API contract.
 *
 * This module owns the 3 private static globals (log file handle, enabled
 * flag, mutex) and the core write path.  All trace-log consumers in the
 * codebase call mglTraceLogExternal (or mglTraceLog in MGLRenderer.m),
 * which funnels through mglTraceLogV under a mutex.
 *
 * Console/stderr output is opt-in via MGL_TRACE_LOG_STDERR=1.  This keeps
 * regular application logs readable while preserving a single trace stream.
 */

#import "mgl_trace_log.h"

#import <Foundation/Foundation.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <dlfcn.h>
#include <limits.h>
#include <pthread.h>
#include <unistd.h>
#include <libgen.h>

/* === Private static globals === */

static FILE *g_mglTraceLogFile = NULL;
static BOOL g_mglTraceLogEnabled = NO;
static BOOL g_mglTraceLogMirrorStderr = NO;
static pthread_mutex_t g_mglTraceLogMutex = PTHREAD_MUTEX_INITIALIZER;

/* === Private env-flag parser (copy of MGLRenderer.m's mglEnvFlagEnabled) ===
 *
 * Kept private to avoid a reverse dependency on the renderer module.
 * MGLRenderer.m's mglEnvFlagEnabled is shared with non-trace code paths
 * (ICB/MTL4 compiler switches) and must stay there. */

static BOOL mglTraceEnvFlag(const char *name)
{
    const char *value = name ? getenv(name) : NULL;
    if (!value || value[0] == '\0') {
        return NO;
    }
    if (strcmp(value, "0") == 0 ||
        strcasecmp(value, "false") == 0 ||
        strcasecmp(value, "no") == 0 ||
        strcasecmp(value, "off") == 0) {
        return NO;
    }
    return YES;
}

/* === Core implementation === */

BOOL mglTraceEnvFlagEnabled(const char *name)
{
    return mglTraceEnvFlag(name);
}

void mglInitTraceLogIfNeeded(void)
{
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        g_mglTraceLogEnabled = mglTraceEnvFlag("MGL_TRACE_LOG");
        g_mglTraceLogMirrorStderr = mglTraceEnvFlag("MGL_TRACE_LOG_STDERR");
        if (!g_mglTraceLogEnabled) {
            return;
        }

        Dl_info info;
        char dylibPath[PATH_MAX] = {0};
        if (dladdr((const void *)&mglInitTraceLogIfNeeded, &info) != 0 &&
            info.dli_fname &&
            info.dli_fname[0] != '\0') {
            snprintf(dylibPath, sizeof(dylibPath), "%s", info.dli_fname);
        } else {
            snprintf(dylibPath, sizeof(dylibPath), ".");
        }

        char dirPath[PATH_MAX] = {0};
        snprintf(dirPath, sizeof(dirPath), "%s", dylibPath);
        char *dirName = dirname(dirPath);
        if (!dirName || dirName[0] == '\0') {
            dirName = ".";
        }

        char logPath[PATH_MAX] = {0};
        snprintf(logPath,
                 sizeof(logPath),
                 "%s/mgl-trace-%d.log",
                 dirName,
                 (int)getpid());

        g_mglTraceLogFile = fopen(logPath, "a");
        if (!g_mglTraceLogFile) {
            fprintf(stderr,
                    "MGL TRACE LOG open failed path=%s errno=%d (%s)\n",
                    logPath,
                    errno,
                    strerror(errno));
            g_mglTraceLogEnabled = NO;
            return;
        }

        setvbuf(g_mglTraceLogFile, NULL, _IOLBF, 0);
        fprintf(g_mglTraceLogFile,
                "MGL TRACE LOG begin pid=%d dylib=%s log=%s built=%s %s\n",
                (int)getpid(),
                info.dli_fname ? info.dli_fname : "(unknown)",
                logPath,
                __DATE__,
                __TIME__);
        fflush(g_mglTraceLogFile);
        if (g_mglTraceLogMirrorStderr) {
            fprintf(stderr, "MGL TRACE LOG enabled path=%s\n", logPath);
        }
    });
}

BOOL mglTraceLogIsEnabled(void)
{
    mglInitTraceLogIfNeeded();
    return g_mglTraceLogEnabled && g_mglTraceLogFile;
}

static void mglTraceLogV(const char *fmt, va_list args)
{
    if (!mglTraceLogIsEnabled() || !fmt) {
        return;
    }

    pthread_mutex_lock(&g_mglTraceLogMutex);
    if (g_mglTraceLogFile) {
        /* The trace log file is opened with _IOLBF (line-buffered),
         * so fputc('\n') already triggers a kernel-level flush.  The
         * explicit fflush was redundant and added a syscall-equivalent
         * overhead per trace line inside the METAL_LOCK. */
        va_list fileArgs;
        va_copy(fileArgs, args);
        vfprintf(g_mglTraceLogFile, fmt, fileArgs);
        va_end(fileArgs);
        fputc('\n', g_mglTraceLogFile);
    }
    if (g_mglTraceLogMirrorStderr) {
        va_list stderrArgs;
        va_copy(stderrArgs, args);
        vfprintf(stderr, fmt, stderrArgs);
        va_end(stderrArgs);
        fputc('\n', stderr);
        fflush(stderr);
    }
    pthread_mutex_unlock(&g_mglTraceLogMutex);
}

void mglTraceLog(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    mglTraceLogV(fmt, args);
    va_end(args);
}

void mglTraceLogExternal(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    mglTraceLogV(fmt, args);
    va_end(args);
}

#ifdef __OBJC__
void mglTraceLogNSStringV(NSString *format, va_list args)
{
    if (!mglTraceLogIsEnabled() || !format) {
        return;
    }

    NSString *message = [[NSString alloc] initWithFormat:format arguments:args];
    const char *utf8 = [message UTF8String];
    mglTraceLog("%s", utf8 ? utf8 : "");
}

void mglTraceLogNSString(NSString *format, ...)
{
    va_list args;
    va_start(args, format);
    mglTraceLogNSStringV(format, args);
    va_end(args);
}
#endif
