/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

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
 * mgl_types_sync.h
 * MGL
 *
 * Sync object type definitions split from glm_context.h.
 */

#ifndef mgl_types_sync_h
#define mgl_types_sync_h

#include "glm_params.h"

typedef struct __GLsync {
    GLsizei name;
    void *mtl_event;
    /* Retained Metal command buffer capturing all GL commands issued before the
     * fence insertion point. The C++ sync path blocks on its completion. Stored as
     * void* (CFBridgingRetain/Release) since this struct is used from plain C. */
    void *mtl_command_buffer;
    /* reference count for deferred sync lifetime management.
     * - newSync sets refcount=1 (caller's GLsync handle)
     * - mglClientWaitSync/mglWaitSync retain at entry, release at exit, so a
     *   glDeleteSync during a concurrent wait cannot free the sync out from
     *   under the waiter
     * - mglDeleteSync sets delete_status and releases; if refcount>0 (wait in
     *   progress), the shell survives until the last release frees it */
#if defined(MGL_GTEST_BUILD)
    /* Host gtests never touch Sync objects; plain int avoids C++/atomic macros. */
    int refcount;
#else
    /* Prefer the _Atomic keyword — do not include <stdatomic.h> here; its
     * atomic_* macros collide with libc++ <atomic> in C++ TUs. */
    _Atomic int refcount;
#endif
    GLboolean delete_status;
#ifdef __cplusplus
} Sync;
#else
} Sync, *__GLsync;
#endif

#endif /* mgl_types_sync_h */
