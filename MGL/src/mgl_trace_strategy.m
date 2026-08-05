/*
 * mgl_trace_strategy.m
 * MGL
 *
 * Implementation of the Trace Strategy Subsystem.
 * See mgl_trace_strategy.h for the architectural rationale.
 *
 * This module is pure policy: it classifies and formats trace records but
 * does not own any mutable state.  The MGLFragmentTextureTraceBinding array
 * and the focus-program list are owned by MGLRenderer.m; this module only
 * reads them.
 */

#import "mgl_trace_strategy.h"

#include <stdlib.h>
#include <string.h>
#include <strings.h>

/* === Fragment texture trace binding === */

BOOL mglFragmentTextureTraceBindingIsInteresting(const MGLFragmentTextureTraceBinding *binding)
{
    return binding &&
           (binding->rt_write_version != 0u ||
            binding->sampled_write_version != 0u ||
            binding->used_sampled_copy ||
            binding->used_fallback ||
            binding->sampled_copy_ptr != NULL);
}

BOOL mglFragmentTextureTraceBindingsHaveInterestingState(const MGLFragmentTextureTraceBinding *bindings,
                                                         NSUInteger count)
{
    if (!bindings) {
        return NO;
    }

    for (NSUInteger i = 0; i < count; i++) {
        if (mglFragmentTextureTraceBindingIsInteresting(&bindings[i])) {
            return YES;
        }
    }

    return NO;
}

BOOL mglFragmentTextureTraceBindingsUseRTSampledCopy(const MGLFragmentTextureTraceBinding *bindings,
                                                     NSUInteger count)
{
    if (!bindings) {
        return NO;
    }

    for (NSUInteger i = 0; i < count; i++) {
        if (bindings[i].used_sampled_copy) {
            return YES;
        }
    }

    return NO;
}

void mglClearFragmentTextureTraceFunctionalFlags(MGLFragmentTextureTraceBinding *bindings,
                                                  NSUInteger count)
{
    if (!bindings || count == 0) {
        return;
    }
    /* Only clear the three fields that non-trace consumers read.
     * used_sampled_copy — checked by mglFragmentTextureTraceBindingsUseRTSampledCopy.
     * rt_write_version  — checked by the batch-replay early-exit (slots 0..3).
     * used_fallback     — checked by mglFragmentTextureTraceBindingIsInteresting
     *                     (only called from trace-gated paths, but clearing it
     *                     here keeps the early-exit fast path consistent). */
    for (NSUInteger i = 0; i < count; i++) {
        bindings[i].used_sampled_copy = 0u;
        bindings[i].rt_write_version = 0u;
        bindings[i].used_fallback = 0u;
    }
}

void mglTraceFragmentTextureTraceBindings(const char *tag,
                                          const char *reason,
                                          const MGLFragmentTextureTraceBinding *bindings,
                                          NSUInteger count,
                                          GLuint program,
                                          GLuint pipelineProgram)
{
    if (!mglTraceLogIsEnabled() ||
        !mglFragmentTextureTraceBindingsHaveInterestingState(bindings, count)) {
        return;
    }

    const MGLFragmentTextureTraceBinding zero = {0};
    const MGLFragmentTextureTraceBinding *slots[4] = {
        count > 0 ? &bindings[0] : &zero,
        count > 1 ? &bindings[1] : &zero,
        count > 2 ? &bindings[2] : &zero,
        count > 3 ? &bindings[3] : &zero
    };

    /* Format each slot's key=value cluster into a per-slot scratch buffer,
     * then join them into the single trace line.  Formatting is
     * trace-gated, so the per-slot sprintf cost is paid only when trace
     * logging is enabled. */
    char slotLines[4][192];
    for (size_t i = 0; i < 4; i++) {
        const MGLFragmentTextureTraceBinding *s = slots[i];
        snprintf(slotLines[i], sizeof(slotLines[i]),
                 "s%zu(tex=%u unit=%u prog=%u mtl=%p direct=%p copy=%p useCopy=%u fallback=%u rtVer=%u sampledVer=%u size=%lux%lu fmt=%lu type=%lu)",
                 i,
                 (unsigned)s->gl_texture_name,
                 (unsigned)s->sampler_unit,
                 (unsigned)s->program_name,
                 s->mtl_texture_ptr,
                 s->direct_mtl_texture_ptr,
                 s->sampled_copy_ptr,
                 (unsigned)s->used_sampled_copy,
                 (unsigned)s->used_fallback,
                 (unsigned)s->rt_write_version,
                 (unsigned)s->sampled_write_version,
                 (unsigned long)s->width,
                 (unsigned long)s->height,
                 (unsigned long)s->pixel_format,
                 (unsigned long)s->texture_type);
    }

    mglTraceLogCategory(MGL_TRACE_CAT_BINDING,
                        "RT_SAMPLE_COPY_SLOTS_%s reason=%s program=%u pipelineProgram=%u %s %s %s %s",
                        tag ? tag : "STATE",
                        reason ? reason : "",
                        (unsigned)program,
                        (unsigned)pipelineProgram,
                        slotLines[0],
                        slotLines[1],
                        slotLines[2],
                        slotLines[3]);
}

/* === Program trace gating === */

bool mglTraceLogProgramListContains(GLuint programName)
{
    if (programName == 0u) {
        return false;
    }

    const char *list = getenv("MGL_TRACE_LOG_PROGRAMS");
    if (!list || list[0] == '\0') {
        return false;
    }

    const char *p = list;
    while (*p != '\0') {
        while (*p == ' ' || *p == '\t' || *p == '\n' ||
               *p == ',' || *p == ';' || *p == ':') {
            p++;
        }
        if (*p == '\0') {
            break;
        }

        char *end = NULL;
        unsigned long value = strtoul(p, &end, 0);
        if (end == p) {
            while (*p != '\0' && *p != ',' && *p != ';' &&
                   *p != ':' && *p != ' ' && *p != '\t' && *p != '\n') {
                p++;
            }
            continue;
        }

        if (value == (unsigned long)programName) {
            return true;
        }
        p = end;
    }

    return false;
}

bool mglProgramExplicitlyTraced(Program *program)
{
    return mglTraceLogIsEnabled() &&
           program &&
           mglTraceLogProgramListContains(program->name);
}

bool mglProgramNeedsTraceLog(Program *program)
{
    return mglTraceLogIsEnabled() &&
           program &&
           (mglProgramExplicitlyTraced(program) ||
            mglProgramNeedsBindingTrace(program));
}

bool mglTraceLogDrawAll(void)
{
    return mglTraceLogIsEnabled() && mglTraceEnvFlagEnabled("MGL_TRACE_LOG_DRAW");
}

bool mglTraceLogResourcesVerbose(void)
{
    return mglTraceLogIsEnabled() && mglTraceEnvFlagEnabled("MGL_TRACE_LOG_RESOURCES");
}

bool mglShouldLogFocusedBinding(uint64_t *counter)
{
    if (!counter) {
        return false;
    }

    uint64_t hit = ++(*counter);
    return hit <= 96ull || (hit % 512ull) == 0ull;
}

bool mglShouldLogTraceFileBindingForProgram(Program *program, uint64_t *counter)
{
    if (!mglTraceLogIsEnabled()) {
        return false;
    }
    if (mglTraceLogResourcesVerbose() ||
        mglProgramExplicitlyTraced(program)) {
        return true;
    }
    return mglShouldLogFocusedBinding(counter);
}
