/*
 * mgl_trace_strategy.h
 * MGL
 *
 * Trace Strategy Subsystem: policy-layer helpers built on top of the
 * trace-log core infrastructure (mgl_trace_log.h).
 *
 * This module owns two groups of trace-strategy decisions:
 *
 *   1. Program trace gating — decides whether a given Program deserves
 *      trace-log output based on env-var allow-lists (MGL_TRACE_LOG_PROGRAMS,
 *      MGL_TRACE_LOG_DRAW, MGL_TRACE_LOG_RESOURCES) and binding-trace
 *      heuristics (mglProgramNeedsBindingTrace from mgl_sampler_compat.h).
 *      These are pure functions over Program metadata + env vars.
 *
 *   2. Fragment texture trace binding — formats per-slot RT_SAMPLE_COPY
 *      diagnostic records (MGLFragmentTextureTraceBinding) for the trace
 *      log.  The binding records themselves are populated by MGLRenderer.m
 *      (which owns the ivar array); this module only classifies and
 *      formats them.
 *
 * What stays in MGLRenderer.m:
 *   - mglTraceResolveDrawProgram / mglTraceShouldLogReplay — depend on
 *     renderer-internal state resolvers (mglResolveProgramFromState,
 *     mglCurrentRenderProgramKey, mglIsFocusedLoadingProgram).
 *   - mglTraceRTYFlipDiagnosticsEnabled / mglYFlipDecisionName — tiny
 *     helpers with few call sites.
 *
 * Dependencies:
 *   - mgl_trace_log.h (core infra: mglTraceLogIsEnabled, mglTraceLog,
 *     mglTraceEnvFlagEnabled)
 *   - glm_context.h (Program type)
 *   - mgl_sampler_compat.h (mglProgramNeedsBindingTrace)
 */

#ifndef MGL_TRACE_STRATEGY_H
#define MGL_TRACE_STRATEGY_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __OBJC__
#endif

#include "glm_context.h"
#include "mgl_trace_log.h"
#include "mgl_sampler_compat.h"

#ifdef __cplusplus
extern "C" {
#endif

/* === Fragment texture trace binding === */

/* Per-slot diagnostic record for RT_SAMPLE_COPY binding decisions.
 * Populated by MGLRenderer.m during fragment-stage texture binding;
 * classified and formatted by this module. */
typedef struct MGLFragmentTextureTraceBinding_t {
    GLuint gl_texture_name;
    GLuint sampler_unit;
    GLuint metal_binding;
    GLuint program_name;
    GLuint rt_write_version;
    GLuint sampled_write_version;
    void *gl_texture_ptr;
    void *mtl_texture_ptr;
    void *direct_mtl_texture_ptr;
    void *sampled_copy_ptr;
    size_t width;
    size_t height;
    size_t pixel_format;
    size_t texture_type;
    uint8_t used_sampled_copy;
    uint8_t used_fallback;
} MGLFragmentTextureTraceBinding;

/* Returns true if the binding record has any non-default state worth tracing. */
bool mglFragmentTextureTraceBindingIsInteresting(const MGLFragmentTextureTraceBinding *binding);

/* Returns true if any binding in the array has interesting state. */
bool mglFragmentTextureTraceBindingsHaveInterestingState(const MGLFragmentTextureTraceBinding *bindings,
                                                         size_t count);

/* Returns true if any binding in the array used a sampled copy. */
bool mglFragmentTextureTraceBindingsUseRTSampledCopy(const MGLFragmentTextureTraceBinding *bindings,
                                                     size_t count);

/* Emits a trace-log line summarizing up to 4 fragment texture trace bindings.
 * No-op when trace logging is disabled or no binding has interesting state. */
void mglTraceFragmentTextureTraceBindings(const char *tag,
                                          const char *reason,
                                          const MGLFragmentTextureTraceBinding *bindings,
                                          size_t count,
                                          GLuint program,
                                          GLuint pipelineProgram);

/* Clears only the functional flag fields (used_sampled_copy,
 * rt_write_version, used_fallback) that non-trace consumers read —
 * mglFragmentTextureTraceBindingsUseRTSampledCopy and the batch-replay
 * early-exit check.  Use this instead of a full-struct memset when trace
 * logging is disabled: it touches ~384 bytes (3 fields × 128 slots) vs
 * ~12 KB for the full array zero. */
void mglClearFragmentTextureTraceFunctionalFlags(MGLFragmentTextureTraceBinding *bindings,
                                                  size_t count);

/* === Program trace gating === */

/* Returns true if programName appears in the MGL_TRACE_LOG_PROGRAMS env var
 * (comma/space/semicolon/colon-separated list).  Pure parse, no state. */
bool mglTraceLogProgramListContains(GLuint programName);

/* Returns true if trace logging is enabled AND program is in the explicit
 * MGL_TRACE_LOG_PROGRAMS list. */
bool mglProgramExplicitlyTraced(Program *program);

/* Returns true if trace logging is enabled AND program is either explicitly
 * traced or matches the binding-trace heuristics (ChunkSection/Sampler1/...). */
bool mglProgramNeedsTraceLog(Program *program);

/* Returns true if MGL_TRACE_LOG_DRAW env var is set (trace all draws). */
bool mglTraceLogDrawAll(void);

/* Returns true if MGL_TRACE_LOG_RESOURCES env var is set (verbose resource
 * binding logs). */
bool mglTraceLogResourcesVerbose(void);

/* Rate-limited focus counter: returns true for the first 96 calls, then
 * every 512th call.  Used to avoid flooding the trace log. */
bool mglShouldLogFocusedBinding(uint64_t *counter);

/* Composite gate: trace if verbose mode is on, program is explicitly traced,
 * or the focus counter says it's time to log. */
bool mglShouldLogTraceFileBindingForProgram(Program *program, uint64_t *counter);

/* === Trace helpers that live in MGLRenderer.m (renderer-state readers) ===
 * Both are plain C functions whose only "ObjC" trait was where they were
 * declared; the declarations moved here so C translation units (the batch
 * replay trace driver) can call them. */

/* Which Program a replay trace line should report (focused/loading program
 * resolution); NULL when nothing is bound. */
Program *mglTraceResolveDrawProgram(GLMContext traceCtx);

/* Whether replay tracing applies to this draw program (env allow-list plus the
 * focus heuristic). */
bool mglTraceShouldLogReplay(GLMContext traceCtx, Program *program);

/* Dump a Program's MSL once (or once per force-reason), when trace logging is
 * on.  `reason` is a C string; reasons mentioning "tex" force the dump. */
void mglWriteProgramMSLDump(Program *program, const char *reason);

/* Texture an attachment renders into, for the trace log. */
Texture *mglTraceFramebufferAttachmentTexture(GLMContext glctx,
                                              FBOAttachment *attachment);

/* Per-vertex attribute sample dump for one replay command.  forceTrace
 * overrides the rate limit. */
void mglTraceReplayCommandVertexAttribSamples(GLMContext traceCtx,
                                              Program *program,
                                              const MGLDrawCommand *cmd,
                                              Buffer *ebo, uint64_t flushId,
                                              uint32_t batchIndex,
                                              uint32_t commandIndex,
                                              bool forceTrace);

#ifdef __cplusplus
}
#endif

#endif /* MGL_TRACE_STRATEGY_H */
