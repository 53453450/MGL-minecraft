/*
 * Copyright (C) Michael Larson on 1/6/2022
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * compute.c
 * MGL
 *
 */

#include "glm_context.h"
#include "draw_command.h"
#include "mgl_safety.h"
#include <string.h>

static GLboolean mglReadDispatchIndirectGroups(Buffer *buf,
                                               GLintptr indirect,
                                               GLuint groups[3])
{
    const uint8_t *base;
    size_t offset;

    if (!buf || !groups || indirect < 0) {
        return GL_FALSE;
    }

    offset = (size_t)indirect;
    if (!buf->data.buffer_data ||
        buf->data.buffer_size == 0u ||
        offset > buf->data.buffer_size ||
        (3u * sizeof(GLuint)) > buf->data.buffer_size - offset) {
        return GL_FALSE;
    }

    base = (const uint8_t *)(uintptr_t)buf->data.buffer_data;
    if (!mglPointerRangeIsReadable(base + offset, 3u * sizeof(GLuint))) {
        return GL_FALSE;
    }

    memcpy(groups, base + offset, 3u * sizeof(GLuint));
    return GL_TRUE;
}

void mglDispatchCompute(GLMContext ctx, GLuint num_groups_x, GLuint num_groups_y, GLuint num_groups_z)
{
    if (!ctx)
        return;

    ERROR_CHECK_RETURN(num_groups_x <= (GLuint)ctx->state.var.max_compute_work_group_count[0], GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(num_groups_y <= (GLuint)ctx->state.var.max_compute_work_group_count[1], GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(num_groups_z <= (GLuint)ctx->state.var.max_compute_work_group_count[2], GL_INVALID_VALUE);

    if (mglShouldSkipConditionalRender(ctx))
        return;

    mglFlushCommandBuffer(ctx);
    ctx->mtl_funcs.mtlDispatchCompute(ctx, num_groups_x, num_groups_y, num_groups_z);
}

void mglDispatchComputeIndirect(GLMContext ctx, GLintptr indirect)
{
    Buffer *buf;
    GLuint groups[3] = {0u, 0u, 0u};

    if (!ctx)
        return;

    if (mglShouldSkipConditionalRender(ctx))
        return;

    if (indirect < 0)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if ((indirect & 0x3) != 0)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    buf = ctx->state.buffers[_DISPATCH_INDIRECT_BUFFER];
    if (ctx->state.var.dispatch_indirect_buffer_binding == 0 || !buf)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    if (buf->mapped && !(buf->access_flags & GL_MAP_PERSISTENT_BIT))
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    if (buf->size < 0 || indirect > buf->size || (GLsizeiptr)(3u * sizeof(GLuint)) > buf->size - indirect)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    if (!mglReadDispatchIndirectGroups(buf, indirect, groups))
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    if ((ctx->state.var.max_compute_work_group_count[0] > 0 &&
         groups[0] > (GLuint)ctx->state.var.max_compute_work_group_count[0]) ||
        (ctx->state.var.max_compute_work_group_count[1] > 0 &&
         groups[1] > (GLuint)ctx->state.var.max_compute_work_group_count[1]) ||
        (ctx->state.var.max_compute_work_group_count[2] > 0 &&
         groups[2] > (GLuint)ctx->state.var.max_compute_work_group_count[2]))
    {
    /* GL 4.6 §7.12.9: an indirect count outside the compute work-group
     * limit is GL_INVALID_VALUE; report it instead of silently skipping. */
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    mglFlushCommandBuffer(ctx);
    ctx->mtl_funcs.mtlDispatchComputeIndirect(ctx, indirect);
}
