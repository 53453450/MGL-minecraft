/*
 * msl_patch_pipeline.h
 * MGL
 *
 * MSL Patch Pipeline Subsystem.
 *
 * A registered pipeline of MSL post-processing steps applied after
 * SPIRV-Cross generates the initial MSL source.  Each step is a named
 * function that may rewrite the MSL string in-place.  The pipeline provides:
 *
 *   - Ordered execution: steps run in registration order; ordering matters
 *     (e.g. sampler rename must precede resource-binding sync).
 *   - Per-step rollback: before each step, the pipeline snapshots the MSL.
 *     If the step returns GL_FALSE, the snapshot is restored so the next
 *     step sees the pre-failure MSL.  A warning is logged.
 *   - Debuggability: each step has a name; failed_step index is recorded.
 *
 * Two pipeline flavors:
 *   - Per-stage pipeline: operates on a single stage's MSL string.
 *     Used in parseSPIRVShaderToMetal (steps 1-17).
 *   - Post-link pipeline: operates on the whole program (may touch multiple
 *     stages).  Used in mglLinkProgram after all stages are compiled.
 *
 * Dependencies: glcorearb.h (GLboolean) + glm_context.h (Program).
 */

#ifndef MSL_PATCH_PIPELINE_H
#define MSL_PATCH_PIPELINE_H

#include "glcorearb.h"
#include "glm_context.h"

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* === Per-stage pipeline === */

/* Context passed to each per-stage patch function. */
typedef struct MSLPatchContext {
    Program *program;
    int stage;
} MSLPatchContext;

/* Per-stage patch function signature.
 *
 * `ctx` carries the Program pointer and shader stage.
 * `msl_ptr` points to the current MSL string (owned by the pipeline).
 * The function may free `*msl_ptr` and replace it with a new malloc'd
 * string.  Must NOT free `*msl_ptr` without replacing it.
 *
 * Returns GL_TRUE on success, GL_FALSE on failure (pipeline will roll back
 * to the pre-step MSL snapshot). */
typedef GLboolean (*MSLPatchFn)(MSLPatchContext *ctx, char **msl_ptr);

/* A single step in the pipeline. */
typedef struct MSLPatchStep {
    const char *name;      /* for logging, must be a string literal */
    MSLPatchFn patch_fn;   /* the patch function */
    GLboolean enabled;     /* GL_FALSE = skip (for debugging) */
} MSLPatchStep;

/* Per-stage pipeline state. */
typedef struct MSLPatchPipeline {
    MSLPatchStep *steps;
    int count;
    int capacity;
    char *msl;             /* current MSL string (owned by pipeline) */
    MSLPatchContext ctx;   /* context passed to each step */
    int failed_step;       /* index of first failed step, or -1 */
} MSLPatchPipeline;

/* Initializes a pipeline.  Takes ownership of `initialMSL` — caller must NOT
 * free it.  `program` and `stage` are stored in the context for patch
 * functions.  Returns GL_FALSE on allocation failure. */
GLboolean mslPipelineInit(MSLPatchPipeline *pipeline,
                          Program *program,
                          int stage,
                          char *initialMSL);

/* Appends a step to the pipeline.  `name` must be a string literal (not
 * copied).  Returns GL_FALSE on allocation failure. */
GLboolean mslPipelineAddStep(MSLPatchPipeline *pipeline,
                             const char *name,
                             MSLPatchFn fn);

/* Runs all enabled steps in order.  Before each step, the pipeline
 * snapshots the current MSL.  If a step returns GL_FALSE, the snapshot is
 * restored (the failed step's changes are discarded) and a warning is
 * logged; `failed_step` is set to that step's index.  Execution continues
 * with the next step so that independent patches are still applied.
 *
 * Returns GL_FALSE if any step failed (even if later steps succeeded),
 * GL_TRUE if all enabled steps succeeded. */
GLboolean mslPipelineRun(MSLPatchPipeline *pipeline);

/* Transfers ownership of the MSL string to the caller.  After this call,
 * the pipeline no longer owns the string and mslPipelineDestroy will not
 * free it.  Returns NULL if the pipeline has no MSL (already taken or
 * never initialized). */
char *mslPipelineTakeResult(MSLPatchPipeline *pipeline);

/* Frees the pipeline's internal resources (steps array and MSL string if
 * not taken).  Does NOT free the Program pointer in the context. */
void mslPipelineDestroy(MSLPatchPipeline *pipeline);

/* === Post-link pipeline === */

/* Post-link patch function signature.  These patches may touch multiple
 * stages' MSL strings (e.g. align FS inputs to VS outputs).  The function
 * reads/writes `program->spirv[stage].msl_str` directly.
 *
 * Returns GL_TRUE on success, GL_FALSE on failure. */
typedef GLboolean (*MSLPatchFnPostLink)(Program *program);

/* A single post-link step. */
typedef struct MSLPatchStepPostLink {
    const char *name;
    MSLPatchFnPostLink patch_fn;
    GLboolean enabled;
} MSLPatchStepPostLink;

/* Post-link pipeline state. */
typedef struct MSLPatchPipelinePostLink {
    MSLPatchStepPostLink *steps;
    int count;
    int capacity;
    Program *program;
    int failed_step;
} MSLPatchPipelinePostLink;

/* Initializes a post-link pipeline.  Does NOT take ownership of `program`. */
GLboolean mslPipelinePostLinkInit(MSLPatchPipelinePostLink *pipeline,
                                  Program *program);

/* Appends a post-link step. */
GLboolean mslPipelinePostLinkAddStep(MSLPatchPipelinePostLink *pipeline,
                                     const char *name,
                                     MSLPatchFnPostLink fn);

/* Runs all enabled post-link steps.  Returns GL_FALSE if any step failed. */
GLboolean mslPipelinePostLinkRun(MSLPatchPipelinePostLink *pipeline);

/* Frees the post-link pipeline's internal resources. */
void mslPipelinePostLinkDestroy(MSLPatchPipelinePostLink *pipeline);

#ifdef __cplusplus
}
#endif

#endif /* MSL_PATCH_PIPELINE_H */
