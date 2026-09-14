/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_stage_copy_back.c — the two copy-back list helpers (P0-1, log 158).
 *
 * Mechanical translation of -clearStageBindingCopyBacks: and
 * -clearStageBindingCopyBack:atIndex:.  `_backend` arrives through the state
 * areas; both methods only touched the backend and the list.
 */

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "mgl_stage_copy_back.h"
#include "mgl_buffer_slots.h"   /* kMGLMaxBufferSlots */
#include "mgl_renderer_ports.h"  /* state areas (backend) */
#include "mgl_renderer_backend.h" /* backend copy-back clear entries */

void mglClearStageBindingCopyBacks(void *renderer,
                                   MGLStageBindingCopyBackList *copy_backs)
{
    if (!copy_backs) {
        return;
    }
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    (void)mglRendererBackendClearStageCopyBackList(areas.backend, copy_backs);
    memset(copy_backs, 0, sizeof(*copy_backs));
}

void mglClearStageBindingCopyBackAtIndex(void *renderer,
                                         MGLStageBindingCopyBackList *copy_backs,
                                         size_t index)
{
    if (!copy_backs || index >= kMGLMaxBufferSlots) {
        return;
    }
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    (void)mglRendererBackendClearStageCopyBackSlot(areas.backend, copy_backs,
                                                   (uint32_t)index);
    MGLStageBindingCopyBack *entry = &copy_backs->slots[index];
    memset(entry, 0, sizeof(*entry));
}
