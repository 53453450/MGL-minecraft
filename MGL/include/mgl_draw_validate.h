/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * Pure drawArrays validation gates (no Metal / no mach).
 */

#ifndef MGL_DRAW_VALIDATE_H
#define MGL_DRAW_VALIDATE_H

#include "glcorearb.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum MGLValidateArraysEarlyStatus {
    MGL_VALIDATE_ARRAYS_OK = 0,
    MGL_VALIDATE_ARRAYS_DISABLED = 1,
    MGL_VALIDATE_ARRAYS_NULL_CTX = 2,
    MGL_VALIDATE_ARRAYS_ZERO_COUNT = 3,
    MGL_VALIDATE_ARRAYS_INVALID_RANGE = 4,
    MGL_VALIDATE_ARRAYS_OVERFLOW = 5
} MGLValidateArraysEarlyStatus;

/* Returns 1 if OK to continue attrib walk, 0 if validation should return. */
int mglDrawValidateArraysEarly(int validation_enabled, int has_ctx, GLint first,
                               GLsizei count, uint64_t *out_first_vertex,
                               uint64_t *out_last_vertex,
                               MGLValidateArraysEarlyStatus *out_status,
                               int *out_ok);

#ifdef __cplusplus
}
#endif

#endif /* MGL_DRAW_VALIDATE_H */
