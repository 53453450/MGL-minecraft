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
#include "mgl_render_pass_manager.h" /* pass-manager transaction entries */
#include "mgl_gpu_recovery.h"       /* guarded call */

/* The .m's file-local command-buffer status enum (MGLRenderer.m). */
enum { MGL_RENDERER_CB_NOT_ENQUEUED = 0u };
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

/* === stage binding copy-back record + flush (P0-1, log 159) ============== */

/* The .m's two file statics, in C. */
static uint64_t mglScbBufferLength(void *buffer)
{
    MGLRenderBufferInfo info = {0};
    return buffer && mglRenderGetBufferInfo(buffer, &info) == 0 ? info.length : 0u;
}

static void mglScbEndBlitEncoder(void *encoder)
{
    (void)mglRenderEndBlitEncoder(encoder);
}

typedef struct {
    void *render_pass_manager;
    void *stage_command_buffer;
    void *recovery_owner;
    MGLRenderCommandBufferTransaction transaction;
    int failed;
} MglScbTransactionCtx;

static int mglScbTransactionGuarded(void *renderer, void *ctx_raw)
{
    (void)renderer;
    MglScbTransactionCtx *c = ctx_raw;
    int rc = mglPassManagerCommitCommandBufferTransaction(
        c->render_pass_manager, c->stage_command_buffer, c->recovery_owner, 1,
        &c->transaction);
    if (rc != 0 || c->transaction.has_error) {
        c->failed = 1;
    }
    return 1;
}

/* -recordStageBindingCopyBack:atIndex:temporary:destination:destination_buffer:
 *  destination_offset:length: */
bool mglRecordStageBindingCopyBack(void *renderer,
                                   MGLStageBindingCopyBackList *copy_backs,
                                   size_t index, void *temporary,
                                   void *destination,
                                   Buffer *destination_buffer,
                                   size_t destination_offset, size_t length)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    if (!copy_backs || index >= kMGLMaxBufferSlots) {
        return false;
    }
    mglClearStageBindingCopyBackAtIndex(renderer, copy_backs, index);
    if (length == 0) {
        return true;
    }
    if (!temporary || !destination ||
        length > mglScbBufferLength(temporary) ||
        destination_offset > mglScbBufferLength(destination) ||
        length > mglScbBufferLength(destination) - destination_offset) {
        return false;
    }

    MGLStageBindingCopyBack *entry = &copy_backs->slots[index];
    if (mglRendererBackendSetStageCopyBackResources(
            areas.backend, copy_backs, (uint32_t)index,
            temporary, destination) != 0) {
        return false;
    }
    entry->temporary = temporary;
    entry->destination = destination;
    entry->destination_buffer = destination_buffer;
    entry->destination_offset = destination_offset;
    entry->length = length;
    return true;}

/* -flushStageBindingCopyBacks:require_cpu_visibility_flag: */
bool mglFlushStageBindingCopyBacks(void *renderer,
                                   MGLStageBindingCopyBackList *copy_backs,
                                   int require_cpu_visibility)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    MGLRendererCoreState *core = areas.core;
    const int require_cpu_visibility_flag = require_cpu_visibility;
    if (!copy_backs) {
        return true;
    }


    MGLRenderCopyBackEntry entries[kMGLMaxBufferSlots];
    memset(entries, 0, sizeof(entries));
    uint32_t entry_count = mglRenderCollectCopyBackEntries(
        (const MGLRenderCopyBackEntry *)copy_backs->slots, kMGLMaxBufferSlots,
        entries, kMGLMaxBufferSlots);
    int has_copies = entry_count > 0u;

    if (mglRenderEncodeStageBindingCopyBacks(
            entries, entry_count, NULL) != 0) {
        mglClearStageBindingCopyBacks(renderer, copy_backs);
        return false;
    }

    if (!has_copies && !require_cpu_visibility_flag) {
        mglClearStageBindingCopyBacks(renderer, copy_backs);
        return true;
    }
    MGLRenderCommandBufferState copy_back_command_state = {0};
    if (!mglRenderCommandBufferOwnerHasState(
            (areas.command ? areas.command->currentCommandBufferOwner : NULL),
            &copy_back_command_state) ||
        copy_back_command_state.status != MGL_RENDERER_CB_NOT_ENQUEUED) {
        mglClearStageBindingCopyBacks(renderer, copy_backs);
        return false;
    }

    if (has_copies) {
        void *blit =
            mglRenderCreateBlitEncoderBorrowed(
                (areas.command ? areas.command->currentCommandBufferOwner : NULL));
        if (!blit) {
            mglClearStageBindingCopyBacks(renderer, copy_backs);
            return false;
        }
        if (mglRenderEncodeStageBindingCopyBacks(
                entries, entry_count, blit) != 0) {
            mglScbEndBlitEncoder(blit);
            mglClearStageBindingCopyBacks(renderer, copy_backs);
            return false;
        }
        mglScbEndBlitEncoder(blit);
    }

    /* Isolated copy-backs must become CPU-visible before another short binding
     * snapshots their destination. TCS also forces this boundary because TES
     * sizing and query accounting currently read its factor buffer on the CPU. */
    void *stage_command_buffer =
        mglPassManagerDetachCurrentCommandBufferForSubmission(areas.render_pass_manager);
    {
        MglScbTransactionCtx tx_ctx = {
            .render_pass_manager = areas.render_pass_manager,
            .stage_command_buffer = stage_command_buffer,
            .recovery_owner = (areas.gpu_recovery_command_owner
                                   ? *areas.gpu_recovery_command_owner
                                   : NULL),
            .transaction = {0},
            .failed = 0,
        };
        if (!mglPlatformShellGuardedCallCtx(
                renderer, "stage binding copy-back transaction",
                mglScbTransactionGuarded, &tx_ctx, NULL)) {
            fprintf(stderr,
                    "MGL BUFFER RANGE: stage synchronization failed: caught "
                    "exception\n");
            mglClearStageBindingCopyBacks(renderer, copy_backs);
            (void)mglRendererNewCommandBufferLockedPort(renderer);
            return false;
        }
        if (tx_ctx.failed) {
            fprintf(stderr,
                    "MGL BUFFER RANGE: C++ stage transaction failed before=%u "
                    "after=%u completion=%u\n",
                    tx_ctx.transaction.before.status,
                    tx_ctx.transaction.after.status,
                    tx_ctx.transaction.completion.status);
            if (tx_ctx.transaction.device_reset_requested) {
                atomic_store_explicit(&core->deviceResetRequested, true,
                                      memory_order_release);
            }
            mglPassManagerReleaseDetachedCommandBufferIfOwned(
                areas.render_pass_manager, stage_command_buffer);
            mglClearStageBindingCopyBacks(renderer, copy_backs);
            (void)mglRendererNewCommandBufferLockedPort(renderer);
            return false;
        }
    }
    MGLRenderCommandBufferState stage_state = {0};
    (void)mglRenderGetCommandBufferState(
        stage_command_buffer, &stage_state);
    if (stage_state.has_error) {
        fprintf(stderr, "MGL BUFFER RANGE: stage command failed: %s\n",
              mglRenderCommandBufferErrorDescription(&stage_state));
        mglClearStageBindingCopyBacks(renderer, copy_backs);
        (mglRendererNewCommandBufferLockedPort(renderer) != 0);
        return false;
    }


    uint32_t failed_index = entry_count;
    if (mglRenderCopyBackCPUPrefix(entries, entry_count, &failed_index) != 0) {
        const MGLRenderCopyBackEntry *failed =
            failed_index < entry_count ? &entries[failed_index] : NULL;
        Buffer *failedBuffer = failed
            ? (Buffer *)(uintptr_t)failed->destination_buffer
            : NULL;
        fprintf(stderr, "MGL BUFFER RANGE: cannot synchronize copied-back prefix to CPU buffer=%u offset=%llu length=%llu cpuSize=%llu\n",
              failedBuffer ? (unsigned)failedBuffer->name : 0u,
              (unsigned long long)(failed ? failed->destination_offset : 0ull),
              (unsigned long long)(failed ? failed->length : 0ull),
              (unsigned long long)(failedBuffer ? failedBuffer->data.buffer_size : 0ull));
        mglClearStageBindingCopyBacks(renderer, copy_backs);
        (mglRendererNewCommandBufferLockedPort(renderer) != 0);
        return false;
    }
    mglClearStageBindingCopyBacks(renderer, copy_backs);
    return (mglRendererNewCommandBufferLockedPort(renderer) != 0);}
