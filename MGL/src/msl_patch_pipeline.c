/*
 * msl_patch_pipeline.c
 * MGL
 *
 * Implementation of the MSL Patch Pipeline Subsystem.
 * See msl_patch_pipeline.h for the API contract.
 *
 * The pipeline owns the MSL string and runs registered patch steps in
 * order.  Before each step, the MSL is snapshotted (strdup).  If the step
 * returns GL_FALSE, the snapshot is restored (the failed step's changes
 * are discarded) and a warning is logged.  This gives per-step rollback
 * semantics that the individual patch functions do not all provide
 * themselves.
 */

#include "msl_patch_pipeline.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* === Per-stage pipeline === */

GLboolean mslPipelineInit(MSLPatchPipeline *pipeline,
                          Program *program,
                          int stage,
                          char *initialMSL)
{
    if (!pipeline) {
        return GL_FALSE;
    }

    pipeline->steps = NULL;
    pipeline->count = 0;
    pipeline->capacity = 0;
    pipeline->msl = initialMSL;  /* take ownership */
    pipeline->ctx.program = program;
    pipeline->ctx.stage = stage;
    pipeline->failed_step = -1;

    return GL_TRUE;
}

GLboolean mslPipelineAddStep(MSLPatchPipeline *pipeline,
                             const char *name,
                             MSLPatchFn fn)
{
    if (!pipeline || !name || !fn) {
        return GL_FALSE;
    }

    if (pipeline->count >= pipeline->capacity) {
        int newCapacity = pipeline->capacity == 0 ? 8 : pipeline->capacity * 2;
        MSLPatchStep *newSteps = (MSLPatchStep *)realloc(
            pipeline->steps,
            (size_t)newCapacity * sizeof(MSLPatchStep));
        if (!newSteps) {
            return GL_FALSE;
        }
        pipeline->steps = newSteps;
        pipeline->capacity = newCapacity;
    }

    pipeline->steps[pipeline->count].name = name;
    pipeline->steps[pipeline->count].patch_fn = fn;
    pipeline->steps[pipeline->count].enabled = GL_TRUE;
    pipeline->count++;

    return GL_TRUE;
}

GLboolean mslPipelineRun(MSLPatchPipeline *pipeline)
{
    if (!pipeline) {
        return GL_FALSE;
    }

    GLboolean allOk = GL_TRUE;

    for (int i = 0; i < pipeline->count; i++) {
        if (!pipeline->steps[i].enabled) {
            continue;
        }

        /* Snapshot MSL before the step for rollback. */
        char *snapshot = NULL;
        if (pipeline->msl) {
            snapshot = strdup(pipeline->msl);
            if (!snapshot) {
                /* Can't snapshot — log and skip the step, keeping current MSL. */
                fprintf(stderr,
                        "MGL MSL PIPELINE: step '%s' skipped (snapshot alloc failed)\n",
                        pipeline->steps[i].name);
                if (pipeline->failed_step < 0) {
                    pipeline->failed_step = i;
                }
                allOk = GL_FALSE;
                continue;
            }
        }

        GLboolean ok = pipeline->steps[i].patch_fn(&pipeline->ctx, &pipeline->msl);

        if (!ok) {
            /* Step failed — roll back to snapshot. */
            fprintf(stderr,
                    "MGL MSL PIPELINE: step '%s' failed, rolling back to pre-step MSL\n",
                    pipeline->steps[i].name);
            if (pipeline->msl) {
                free(pipeline->msl);
            }
            pipeline->msl = snapshot;  /* restore pre-step MSL */
            snapshot = NULL;           /* pipeline owns it now */
            if (pipeline->failed_step < 0) {
                pipeline->failed_step = i;
            }
            allOk = GL_FALSE;
        } else if (pipeline->msl == NULL) {
            /* Step succeeded but nulled the MSL — treat as failure. */
            fprintf(stderr,
                    "MGL MSL PIPELINE: step '%s' left MSL NULL, rolling back\n",
                    pipeline->steps[i].name);
            pipeline->msl = snapshot;
            snapshot = NULL;
            if (pipeline->failed_step < 0) {
                pipeline->failed_step = i;
            }
            allOk = GL_FALSE;
        }

        if (snapshot) {
            free(snapshot);
        }
    }

    return allOk;
}

char *mslPipelineTakeResult(MSLPatchPipeline *pipeline)
{
    if (!pipeline) {
        return NULL;
    }

    char *result = pipeline->msl;
    pipeline->msl = NULL;
    return result;
}

void mslPipelineDestroy(MSLPatchPipeline *pipeline)
{
    if (!pipeline) {
        return;
    }

    if (pipeline->msl) {
        free(pipeline->msl);
        pipeline->msl = NULL;
    }

    if (pipeline->steps) {
        free(pipeline->steps);
        pipeline->steps = NULL;
    }

    pipeline->count = 0;
    pipeline->capacity = 0;
    pipeline->failed_step = -1;
}

/* === Post-link pipeline === */

GLboolean mslPipelinePostLinkInit(MSLPatchPipelinePostLink *pipeline,
                                  Program *program)
{
    if (!pipeline) {
        return GL_FALSE;
    }

    pipeline->steps = NULL;
    pipeline->count = 0;
    pipeline->capacity = 0;
    pipeline->program = program;
    pipeline->failed_step = -1;

    return GL_TRUE;
}

GLboolean mslPipelinePostLinkAddStep(MSLPatchPipelinePostLink *pipeline,
                                     const char *name,
                                     MSLPatchFnPostLink fn)
{
    if (!pipeline || !name || !fn) {
        return GL_FALSE;
    }

    if (pipeline->count >= pipeline->capacity) {
        int newCapacity = pipeline->capacity == 0 ? 8 : pipeline->capacity * 2;
        MSLPatchStepPostLink *newSteps = (MSLPatchStepPostLink *)realloc(
            pipeline->steps,
            (size_t)newCapacity * sizeof(MSLPatchStepPostLink));
        if (!newSteps) {
            return GL_FALSE;
        }
        pipeline->steps = newSteps;
        pipeline->capacity = newCapacity;
    }

    pipeline->steps[pipeline->count].name = name;
    pipeline->steps[pipeline->count].patch_fn = fn;
    pipeline->steps[pipeline->count].enabled = GL_TRUE;
    pipeline->count++;

    return GL_TRUE;
}

GLboolean mslPipelinePostLinkRun(MSLPatchPipelinePostLink *pipeline)
{
    if (!pipeline) {
        return GL_FALSE;
    }

    GLboolean allOk = GL_TRUE;

    for (int i = 0; i < pipeline->count; i++) {
        if (!pipeline->steps[i].enabled) {
            continue;
        }

        GLboolean ok = pipeline->steps[i].patch_fn(pipeline->program);

        if (!ok) {
            fprintf(stderr,
                    "MGL MSL PIPELINE: post-link step '%s' failed\n",
                    pipeline->steps[i].name);
            if (pipeline->failed_step < 0) {
                pipeline->failed_step = i;
            }
            allOk = GL_FALSE;
            /* No rollback for post-link steps — they mutate program state
             * across multiple stages, and snapshotting all stages' MSL is
             * too expensive.  The step's own function should handle its
             * internal rollback. */
        }
    }

    return allOk;
}

void mslPipelinePostLinkDestroy(MSLPatchPipelinePostLink *pipeline)
{
    if (!pipeline) {
        return;
    }

    if (pipeline->steps) {
        free(pipeline->steps);
        pipeline->steps = NULL;
    }

    pipeline->count = 0;
    pipeline->capacity = 0;
    pipeline->failed_step = -1;
}
