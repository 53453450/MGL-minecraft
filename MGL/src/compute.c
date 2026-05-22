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
#include <string.h>


void mglDispatchCompute(GLMContext ctx, GLuint num_groups_x, GLuint num_groups_y, GLuint num_groups_z)
{
    if (!ctx)
        return;

    ERROR_CHECK_RETURN(num_groups_x <= (GLuint)ctx->state.var.max_compute_work_group_count[0], GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(num_groups_y <= (GLuint)ctx->state.var.max_compute_work_group_count[1], GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(num_groups_z <= (GLuint)ctx->state.var.max_compute_work_group_count[2], GL_INVALID_VALUE);

    ctx->mtl_funcs.mtlDispatchCompute(ctx, num_groups_x, num_groups_y, num_groups_z);
}

void mglDispatchComputeIndirect(GLMContext ctx, GLintptr indirect)
{
    Buffer *buf;
    const uint8_t *base;
    GLuint groups[3] = {0u, 0u, 0u};

    if (!ctx)
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
    if (buf->mapped || !buf->data.buffer_data)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    if (buf->size < 0 || indirect > buf->size || (GLsizeiptr)sizeof(groups) > buf->size - indirect)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    base = (const uint8_t *)(uintptr_t)buf->data.buffer_data;
    memcpy(groups, base + indirect, sizeof(groups));

    mglDispatchCompute(ctx, groups[0], groups[1], groups[2]);
}
