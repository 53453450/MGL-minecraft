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
    const MGLFragmentTextureTraceBinding *s0 = count > 0 ? &bindings[0] : &zero;
    const MGLFragmentTextureTraceBinding *s1 = count > 1 ? &bindings[1] : &zero;
    const MGLFragmentTextureTraceBinding *s2 = count > 2 ? &bindings[2] : &zero;
    const MGLFragmentTextureTraceBinding *s3 = count > 3 ? &bindings[3] : &zero;

    mglTraceLog("RT_SAMPLE_COPY_SLOTS_%s reason=%s program=%u pipelineProgram=%u "
                "s0(tex=%u unit=%u prog=%u mtl=%p direct=%p copy=%p useCopy=%u fallback=%u rtVer=%u sampledVer=%u size=%lux%lu fmt=%lu type=%lu) "
                "s1(tex=%u unit=%u prog=%u mtl=%p direct=%p copy=%p useCopy=%u fallback=%u rtVer=%u sampledVer=%u size=%lux%lu fmt=%lu type=%lu) "
                "s2(tex=%u unit=%u prog=%u mtl=%p direct=%p copy=%p useCopy=%u fallback=%u rtVer=%u sampledVer=%u size=%lux%lu fmt=%lu type=%lu) "
                "s3(tex=%u unit=%u prog=%u mtl=%p direct=%p copy=%p useCopy=%u fallback=%u rtVer=%u sampledVer=%u size=%lux%lu fmt=%lu type=%lu)",
                tag ? tag : "STATE",
                reason ? reason : "",
                (unsigned)program,
                (unsigned)pipelineProgram,
                (unsigned)s0->gl_texture_name,
                (unsigned)s0->sampler_unit,
                (unsigned)s0->program_name,
                s0->mtl_texture_ptr,
                s0->direct_mtl_texture_ptr,
                s0->sampled_copy_ptr,
                (unsigned)s0->used_sampled_copy,
                (unsigned)s0->used_fallback,
                (unsigned)s0->rt_write_version,
                (unsigned)s0->sampled_write_version,
                (unsigned long)s0->width,
                (unsigned long)s0->height,
                (unsigned long)s0->pixel_format,
                (unsigned long)s0->texture_type,
                (unsigned)s1->gl_texture_name,
                (unsigned)s1->sampler_unit,
                (unsigned)s1->program_name,
                s1->mtl_texture_ptr,
                s1->direct_mtl_texture_ptr,
                s1->sampled_copy_ptr,
                (unsigned)s1->used_sampled_copy,
                (unsigned)s1->used_fallback,
                (unsigned)s1->rt_write_version,
                (unsigned)s1->sampled_write_version,
                (unsigned long)s1->width,
                (unsigned long)s1->height,
                (unsigned long)s1->pixel_format,
                (unsigned long)s1->texture_type,
                (unsigned)s2->gl_texture_name,
                (unsigned)s2->sampler_unit,
                (unsigned)s2->program_name,
                s2->mtl_texture_ptr,
                s2->direct_mtl_texture_ptr,
                s2->sampled_copy_ptr,
                (unsigned)s2->used_sampled_copy,
                (unsigned)s2->used_fallback,
                (unsigned)s2->rt_write_version,
                (unsigned)s2->sampled_write_version,
                (unsigned long)s2->width,
                (unsigned long)s2->height,
                (unsigned long)s2->pixel_format,
                (unsigned long)s2->texture_type,
                (unsigned)s3->gl_texture_name,
                (unsigned)s3->sampler_unit,
                (unsigned)s3->program_name,
                s3->mtl_texture_ptr,
                s3->direct_mtl_texture_ptr,
                s3->sampled_copy_ptr,
                (unsigned)s3->used_sampled_copy,
                (unsigned)s3->used_fallback,
                (unsigned)s3->rt_write_version,
                (unsigned)s3->sampled_write_version,
                (unsigned long)s3->width,
                (unsigned long)s3->height,
                (unsigned long)s3->pixel_format,
                (unsigned long)s3->texture_type);
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
