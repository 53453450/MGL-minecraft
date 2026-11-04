/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * Pure drawArrays validation gates (no Metal). HostOps runner lives in
 * mgl_draw_issue.cpp.
 */

#include "mgl_draw_validate.h"

#include <stdint.h>

int mglDrawValidateArraysEarly(int validation_enabled, int has_ctx, GLint first,
                               GLsizei count, uint64_t *out_first_vertex,
                               uint64_t *out_last_vertex,
                               MGLValidateArraysEarlyStatus *out_status,
                               int *out_ok)
{
    if (out_first_vertex) {
        *out_first_vertex = 0u;
    }
    if (out_last_vertex) {
        *out_last_vertex = 0u;
    }
    if (out_ok) {
        *out_ok = 0;
    }
    if (!validation_enabled) {
        if (out_status) {
            *out_status = MGL_VALIDATE_ARRAYS_DISABLED;
        }
        if (out_ok) {
            *out_ok = 1;
        }
        return 0;
    }
    if (!has_ctx) {
        if (out_status) {
            *out_status = MGL_VALIDATE_ARRAYS_NULL_CTX;
        }
        return 0;
    }
    if (count == 0) {
        if (out_status) {
            *out_status = MGL_VALIDATE_ARRAYS_ZERO_COUNT;
        }
        return 0;
    }
    if (count < 0 || first < 0) {
        if (out_status) {
            *out_status = MGL_VALIDATE_ARRAYS_INVALID_RANGE;
        }
        return 0;
    }
    uint64_t firstVertex = (uint64_t)(uint32_t)first;
    uint64_t vertexCount = (uint64_t)(uint32_t)count;
    if (vertexCount == 0u || firstVertex > UINT64_MAX - (vertexCount - 1u)) {
        if (out_status) {
            *out_status = MGL_VALIDATE_ARRAYS_OVERFLOW;
        }
        return 0;
    }
    if (out_first_vertex) {
        *out_first_vertex = firstVertex;
    }
    if (out_last_vertex) {
        *out_last_vertex = firstVertex + vertexCount - 1u;
    }
    if (out_status) {
        *out_status = MGL_VALIDATE_ARRAYS_OK;
    }
    if (out_ok) {
        *out_ok = 1;
    }
    return 1;
}
