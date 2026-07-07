/*
 * mgl_trace_log.h
 * MGL
 *
 * Trace Log Subsystem: core infrastructure for MGL's diagnostic trace
 * logging.  When the MGL_TRACE_LOG environment variable is enabled, trace
 * messages are written to a per-process log file (mgl-trace-<pid>.log)
 * in the dylib's directory.
 *
 * Public API:
 *   - mglTraceLogExternal(fmt, ...)  — the ONLY function external callers
 *     should use.  Declared here to replace the ~11 scattered `extern`
 *     declarations across the codebase.
 *
 * Internal API (for MGLRenderer.m):
 *   - mglTraceLog(fmt, ...)          — same as mglTraceLogExternal but for
 *     in-renderer call sites that don't want the "External" suffix.
 *   - mglTraceLogIsEnabled()         — gate check (also lazily initializes
 *     the log file via dispatch_once).
 *   - MGLTraceNSLog(fmt, ...)        — legacy ObjC NSString-format wrapper
 *     gated by trace enabled state (static inline, ObjC only).  Despite the
 *     name, it writes to the trace log, not NSLog, unless stderr mirroring is
 *     explicitly enabled.
 *
 * Design notes:
 *   - 3 static globals (log file handle, enabled flag, mutex) are private
 *     to mgl_trace_log.m and never exposed.
 *   - mglTraceLogIsEnabled() is a thin accessor that also triggers lazy
 *     initialization.  Callers that check it frequently pay only a
 *     dispatch_once predicate check after first init.
 *   - The env-flag parser (mglTraceEnvFlag) is a private copy of
 *     MGLRenderer.m's mglEnvFlagEnabled — kept private to avoid a reverse
 *     dependency on the renderer module.
 */

#ifndef MGL_TRACE_LOG_H
#define MGL_TRACE_LOG_H

#include <objc/objc.h>  /* BOOL */
#include <stdarg.h>

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* The sole public trace-logging entry point.  No-op when trace logging is
 * disabled.  Safe to call from any translation unit — replaces the ~11
 * scattered `extern void mglTraceLogExternal(...)` declarations. */
void mglTraceLogExternal(const char *fmt, ...);

/* In-renderer trace log (identical to mglTraceLogExternal).  Provided for
 * call sites within MGLRenderer.m that predate the External/External split. */
void mglTraceLog(const char *fmt, ...);

/* Returns YES if trace logging is enabled and the log file is open.
 * Lazily initializes the log file on first call (dispatch_once). */
BOOL mglTraceLogIsEnabled(void);

/* Trace-specific env-flag parser.  Returns YES if the named environment
 * variable is set to a truthy value (non-empty, non-0/false/no/off).
 * Exposed so trace-strategy modules can query env flags (e.g.
 * MGL_TRACE_LOG_DRAW, MGL_TRACE_LOG_RESOURCES) without depending on
 * MGLRenderer.m's mglEnvFlagEnabled (which is shared with non-trace
 * code paths like ICB/MTL4 compiler switches). */
BOOL mglTraceEnvFlagEnabled(const char *name);

/* Explicitly trigger log-file initialization.  Normally called via
 * mglTraceLogIsEnabled(); exposed for the constructor attribute in
 * MGLRenderer.m. */
void mglInitTraceLogIfNeeded(void);

#ifdef __OBJC__
/* ObjC NSString-format wrapper gated by trace-enabled state.  Kept under the
 * old MGLTraceNSLog name so existing call sites keep compiling while trace
 * output is centralized in mgl-trace-<pid>.log. */
void mglTraceLogNSStringV(NSString *format, va_list args);
void mglTraceLogNSString(NSString *format, ...);

static inline void MGLTraceNSLog(NSString *format, ...) {
    if (mglTraceLogIsEnabled()) {
        va_list args;
        va_start(args, format);
        mglTraceLogNSStringV(format, args);
        va_end(args);
    }
}
#endif

#ifdef __cplusplus
}
#endif

#endif /* MGL_TRACE_LOG_H */
