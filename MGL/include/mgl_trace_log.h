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
 *   - mglTraceLogCategory(cat, ...)   — explicit semantic category.
 *   - mglTraceFrameID() / mglTraceNoteFrameBoundary() — monotonic frame
 *     counter for the fid= prefix field.
 *   - mglTraceLogIsEnabled()         — gate check (also lazily initializes
 *     the log file via dispatch_once).
 *   - mglTraceLogNSString(fmt, ...)  — ObjC NSString-format wrapper gated
 *     by trace enabled state.  Despite the legacy name similarity, it
 *     writes to the trace log, not NSLog, unless stderr mirroring is
 *     explicitly enabled.
 *
 * Design notes:
 *   - 3 static globals (log file handle, enabled flag, mutex) are private
 *     to mgl_trace_log.m and never exposed.
 *   - Every written line is prefixed with
 *     [<mono_ns> <seq> tid=<tid> fid=<frame> cat=<CAT>] by the write path
 *     (mglTraceLogV), so lines carry a monotonic timestamp, a global
 *     sequence number, thread id, frame id, and semantic category without
 *     call sites doing anything.
 *   - The env-flag parser (mglTraceEnvFlag) is a private copy of
 *     MGLRenderer.m's mglEnvFlagEnabled — kept private to avoid a reverse
 *     dependency on the renderer module.
 */

#ifndef MGL_TRACE_LOG_H
#define MGL_TRACE_LOG_H

#include <objc/objc.h>  /* BOOL */
#include <stdarg.h>
#include <stdint.h>
#include <mach/mach_time.h>

/* Monotonic nanosecond clock for trace timing.  Uses mach_absolute_time()
 * (not CFAbsoluteTimeGetCurrent) so elapsed values are immune to wall-clock
 * steps (e.g. NTP).  Returns nanoseconds since an arbitrary epoch. */
static inline uint64_t mglTraceClockNS(void)
{
    static mach_timebase_info_data_t tb = {0, 0};
    if (tb.denom == 0) {
        mach_timebase_info(&tb);
    }
    return (uint64_t)((double)mach_absolute_time() * (double)tb.numer / (double)tb.denom);
}

/* Semantic trace categories (apitrace-style flag bits).  Each written line
 * is tagged with one of these via the cat= field; consumers filter with
 * awk on cat= without parsing the message body. */
typedef enum {
    MGL_TRACE_CAT_DEFAULT = 0,
    MGL_TRACE_CAT_DRAW,       /* DRAW_* / MULTI_DRAW_* / VATTR_* geometry submission */
    MGL_TRACE_CAT_RESOURCE,   /* TEXTURE_* / TEX_* resource lifecycle */
    MGL_TRACE_CAT_PROGRAM,    /* program link / PSO-ish program state */
    MGL_TRACE_CAT_BINDING,    /* RT_SAMPLE_COPY* / TBIND / VBIND / BINDMAP binding decisions */
    MGL_TRACE_CAT_PSO,        /* RENDERPASS_* encoder / pipeline state */
    MGL_TRACE_CAT_SWAP,       /* SWAP_* frame boundary */
    MGL_TRACE_CAT_PERF        /* PERF* counters / elapsed lines */
} MGLTraceCategory;

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* The trace-logging entry point.  No-op when trace logging is disabled.
 * Safe to call from any translation unit.  mglTraceLogExternal is kept as a
 * source-compatible alias: the two functions had byte-identical bodies, so
 * the duplicate implementation was removed (see mgl_trace_log.m). */
void mglTraceLog(const char *fmt, ...);
#define mglTraceLogExternal mglTraceLog

/* Trace-log entry with an explicit semantic category (see MGLTraceCategory).
 * The category appears in the cat= prefix field; mglTraceLog infers a
 * category from the message token when it can, otherwise DEFAULT. */
void mglTraceLogCategory(MGLTraceCategory cat, const char *fmt, ...);

/* Returns YES if trace logging is enabled and the log file is open.
 * Lazily initializes the log file on first call (dispatch_once). */
BOOL mglTraceLogIsEnabled(void);

/* Monotonic frame counter for the fid= prefix field.  Incremented at the
 * swap boundary by the renderer (mglTraceNoteFrameBoundary); falls back to
 * an internal counter when the renderer never calls it. */
uint64_t mglTraceFrameID(void);
void mglTraceNoteFrameBoundary(void);

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
/* ObjC NSString-format wrapper gated by trace-enabled state.  The write
 * path (mglTraceLogV) adds the same [mono_ns seq tid fid cat] prefix as
 * mglTraceLog, so NSString call sites are indistinguishable in the log. */
void mglTraceLogNSStringV(NSString *format, va_list args);
void mglTraceLogNSString(NSString *format, ...);
#endif

#ifdef __cplusplus
}
#endif

#endif /* MGL_TRACE_LOG_H */
