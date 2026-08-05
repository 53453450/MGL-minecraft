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
#include "mgl_env_flag.h"

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
#include <stdatomic.h>

/* === Private static globals === */

static FILE *g_mglTraceLogFile = NULL;
static BOOL g_mglTraceLogEnabled = NO;
static BOOL g_mglTraceLogMirrorStderr = NO;
static pthread_mutex_t g_mglTraceLogMutex = PTHREAD_MUTEX_INITIALIZER;

static _Atomic uint64_t g_mglTraceSeq = 0;
static _Atomic uint64_t g_mglTraceFrameID = 0;
static uint64_t g_mglTraceFallbackFrameID = 0;

/* === Private env-flag parser ===
 * Delegates to the single-source mgl_env_flag_enabled() in mgl_env_flag.h.
 * (Formerly a private copy of MGLRenderer.m's mglEnvFlagEnabled.) */

static BOOL mglTraceEnvFlag(const char *name)
{
    return mgl_env_flag_enabled(name) ? YES : NO;
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
                "MGL TRACE LOG begin format=1 pid=%d dylib=%s log=%s built=%s %s\n",
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

uint64_t mglTraceFrameID(void)
{
    uint64_t frame = atomic_load_explicit(&g_mglTraceFrameID, memory_order_relaxed);
    if (frame == 0) {
        /* No swap boundary observed yet (e.g. trace from a test harness
         * that never swaps); fall back to a monotonically increasing local
         * counter so fid= is still a usable ordering key. */
        return ++g_mglTraceFallbackFrameID;
    }
    return frame;
}

void mglTraceNoteFrameBoundary(void)
{
    atomic_fetch_add_explicit(&g_mglTraceFrameID, 1, memory_order_relaxed);
}

/* Map a trace message's leading token to a semantic category.  Messages
 * that don't match a known prefix get DEFAULT. */
static MGLTraceCategory mglTraceCategoryForFormat(const char *fmt)
{
    if (!fmt) {
        return MGL_TRACE_CAT_DEFAULT;
    }
    if (strncmp(fmt, "DRAW_", 5) == 0 ||
        strncmp(fmt, "MULTI_DRAW_", 11) == 0 ||
        strncmp(fmt, "VATTR_", 6) == 0) {
        return MGL_TRACE_CAT_DRAW;
    }
    if (strncmp(fmt, "TEXTURE_", 8) == 0 || strncmp(fmt, "TEX_", 4) == 0) {
        return MGL_TRACE_CAT_RESOURCE;
    }
    if (strncmp(fmt, "RT_SAMPLE_COPY", 14) == 0 ||
        strncmp(fmt, "TBIND", 5) == 0 ||
        strncmp(fmt, "VBIND", 5) == 0 ||
        strncmp(fmt, "BINDMAP", 7) == 0 ||
        strncmp(fmt, "BINDMISS", 8) == 0) {
        return MGL_TRACE_CAT_BINDING;
    }
    if (strncmp(fmt, "RENDERPASS_", 11) == 0) {
        return MGL_TRACE_CAT_PSO;
    }
    if (strncmp(fmt, "SWAP_", 5) == 0) {
        return MGL_TRACE_CAT_SWAP;
    }
    if (strncmp(fmt, "PERF", 4) == 0) {
        return MGL_TRACE_CAT_PERF;
    }
    return MGL_TRACE_CAT_DEFAULT;
}

static const char *mglTraceCategoryName(MGLTraceCategory cat)
{
    switch (cat) {
        case MGL_TRACE_CAT_DRAW:     return "DRAW";
        case MGL_TRACE_CAT_RESOURCE: return "RESOURCE";
        case MGL_TRACE_CAT_PROGRAM:  return "PROGRAM";
        case MGL_TRACE_CAT_BINDING:  return "BINDING";
        case MGL_TRACE_CAT_PSO:      return "PSO";
        case MGL_TRACE_CAT_SWAP:     return "SWAP";
        case MGL_TRACE_CAT_PERF:     return "PERF";
        default:                     return "DEFAULT";
    }
}

/* Write one fully-formatted line to the trace log (and optionally stderr).
 * The caller has already applied the per-line prefix.  Embedded newlines
 * are escaped so every trace record occupies exactly one line (schema
 * stability: consumers can rely on line == record). */
static void mglTraceLogWriteLine(const char *line)
{
    if (g_mglTraceLogFile) {
        fputs(line, g_mglTraceLogFile);
        fputc('\n', g_mglTraceLogFile);
    }
    if (g_mglTraceLogMirrorStderr) {
        fputs(line, stderr);
        fputc('\n', stderr);
        fflush(stderr);
    }
}

/* Rate-limiter for SKIP-class messages (DRAW_*_SKIP, VATTR_SAMPLE, ...).
 * SKIP lines are diagnostic reasons for draws the PERF counters already
 * track as a count; flooding the log with one line per skip is noise.
 * The first SKIP_APPROVAL_THRESHOLD occurrences of each distinct reason
 * are written in full; after that one line is written per
 * SKIP_APPROVAL_PERIOD and the remaining ones are counted in `dropped`.
 * The dropped count rides along on the periodic line so no information is
 * silently lost. */
#define MGL_TRACE_SKIP_INITIAL_LIMIT 8u
#define MGL_TRACE_SKIP_PERIOD 512u

typedef struct {
    char reason[48];
    uint64_t hit;
    uint64_t dropped;
} MGLTraceSkipSlot;

static MGLTraceSkipSlot g_mglTraceSkipSlots[8];
static uint64_t g_mglTraceSkipSlotCount = 0;

/* Returns true if this SKIP message should be written, false if it is
 * being rate-limited.  When rate-limiting, `dropped_out` receives the
 * total number of dropped lines for this reason (0 when not dropping). */
static bool mglTraceSkipRateLimit(const char *body, uint64_t *dropped_out)
{
    *dropped_out = 0;
    if (!body) {
        return true;
    }

    const char *reason = strstr(body, "reason=");
    if (!reason) {
        return true;
    }
    reason += strlen("reason=");
    const char *end = reason;
    while (*end && *end != ' ' && *end != '\t' && *end != '\n' && *end != '\r') {
        end++;
    }
    size_t reasonLen = (size_t)(end - reason);
    if (reasonLen == 0) {
        return true;
    }
    if (reasonLen >= sizeof(g_mglTraceSkipSlots[0].reason)) {
        reasonLen = sizeof(g_mglTraceSkipSlots[0].reason) - 1;
    }

    MGLTraceSkipSlot *slot = NULL;
    for (uint64_t i = 0; i < g_mglTraceSkipSlotCount; i++) {
        if (strncmp(g_mglTraceSkipSlots[i].reason, reason, reasonLen) == 0 &&
            g_mglTraceSkipSlots[i].reason[reasonLen] == '\0') {
            slot = &g_mglTraceSkipSlots[i];
            break;
        }
    }
    if (!slot && g_mglTraceSkipSlotCount < sizeof(g_mglTraceSkipSlots) / sizeof(g_mglTraceSkipSlots[0])) {
        slot = &g_mglTraceSkipSlots[g_mglTraceSkipSlotCount++];
        memcpy(slot->reason, reason, reasonLen);
        slot->reason[reasonLen] = '\0';
        slot->hit = 0;
        slot->dropped = 0;
    }

    if (!slot) {
        return true;
    }

    uint64_t hit = ++slot->hit;
    if (hit <= MGL_TRACE_SKIP_INITIAL_LIMIT) {
        return true;
    }
    if ((hit % MGL_TRACE_SKIP_PERIOD) == 0ull) {
        *dropped_out = slot->dropped;
        return true;
    }
    slot->dropped++;
    return false;
}

static void mglTraceLogV(MGLTraceCategory cat, const char *fmt, va_list args)
{
    if (!mglTraceLogIsEnabled() || !fmt) {
        return;
    }

    /* Format the body into a bounded stack buffer (no heap churn on the
     * hot path); fall back to a heap buffer for oversized messages. */
    char stackBuf[2048];
    char *body = stackBuf;
    int bodyLen = 0;
    {
        va_list copy;
        va_copy(copy, args);
        bodyLen = vsnprintf(stackBuf, sizeof(stackBuf), fmt, copy);
        va_end(copy);
    }
    if (bodyLen < 0) {
        return;
    }
    if (bodyLen >= (int)sizeof(stackBuf)) {
        body = malloc((size_t)bodyLen + 1);
        if (!body) {
            return;
        }
        va_list copy;
        va_copy(copy, args);
        vsnprintf(body, (size_t)bodyLen + 1, fmt, copy);
        va_end(copy);
    }

    uint64_t seq = atomic_fetch_add_explicit(&g_mglTraceSeq, 1, memory_order_relaxed) + 1;
    uint64_t frame = mglTraceFrameID();
    uint64_t monoNs = mglTraceClockNS();
    uint64_t tid = 0;
    pthread_threadid_np(NULL, &tid);
    if (cat == MGL_TRACE_CAT_DEFAULT) {
        cat = mglTraceCategoryForFormat(fmt);
    }

    /* Rate-limit SKIP-class messages; the PERF counters already carry the
     * aggregate skip count, so per-skip reason lines are flood-prone. */
    uint64_t skipDropped = 0;
    if (!mglTraceSkipRateLimit(body, &skipDropped)) {
        if (body != stackBuf) {
            free(body);
        }
        return;
    }

    char prefix[256];
    int prefixLen = snprintf(prefix,
                             sizeof(prefix),
                             "[%llu %llu tid=%llu fid=%llu cat=%s] ",
                             (unsigned long long)monoNs,
                             (unsigned long long)seq,
                             (unsigned long long)tid,
                             (unsigned long long)frame,
                             mglTraceCategoryName(cat));

    /* Escape the body so no embedded \n / \r breaks the one-line record.
     * Escaped length is at most 2*bodyLen (worst case: every byte is a
     * newline), so the line buffer must be sized accordingly — the
     * previous bodyLen+prefixLen+2 sizing overflowed for newline-heavy
     * bodies and corrupted adjacent heap chunks. */
    size_t escapedLen = 0;
    for (const char *c = body; *c; c++) {
        escapedLen += (*c == '\n' || *c == '\r') ? 2u : 1u;
    }
    size_t lineCap = escapedLen + (size_t)prefixLen + 32u;
    char *line = malloc(lineCap);
    if (!line) {
        if (body != stackBuf) {
            free(body);
        }
        return;
    }
    if (prefixLen > 0) {
        memcpy(line, prefix, (size_t)prefixLen);
    }
    size_t outPos = (size_t)prefixLen;
    for (const char *c = body; *c; c++) {
        if (*c == '\n') {
            line[outPos++] = '\\';
            line[outPos++] = 'n';
        } else if (*c == '\r') {
            line[outPos++] = '\\';
            line[outPos++] = 'r';
        } else {
            line[outPos++] = *c;
        }
    }
    if (skipDropped > 0) {
        int appended = snprintf(line + outPos,
                                lineCap - outPos,
                                " dropped=%llu",
                                (unsigned long long)skipDropped);
        if (appended > 0) {
            outPos += (size_t)appended;
        }
    }
    line[outPos] = '\0';

    pthread_mutex_lock(&g_mglTraceLogMutex);
    if (g_mglTraceLogFile) {
        mglTraceLogWriteLine(line);
    }
    pthread_mutex_unlock(&g_mglTraceLogMutex);

    if (body != stackBuf) {
        free(body);
    }
    free(line);
}

void mglTraceLog(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    mglTraceLogV(MGL_TRACE_CAT_DEFAULT, fmt, args);
    va_end(args);
}

void mglTraceLogCategory(MGLTraceCategory cat, const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    mglTraceLogV(cat, fmt, args);
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
