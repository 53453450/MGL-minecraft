/* SPDX-License-Identifier: LGPL-3.0-only */

#include "mgl_renderer_core_state.h"
#include "mgl_encode_context.h"

#include <stdio.h>
#include <string.h>

static int failures;

#define CHECK(condition, message)                  \
    do {                                           \
        if (!(condition)) {                        \
            fprintf(stderr, "FAIL: %s\n", message); \
            failures++;                            \
        }                                          \
    } while (0)

int main(void)
{
    GLMContextRec ctx;
    GLMState workspace;
    memset(&ctx, 0, sizeof(ctx));
    memset(&workspace, 0, sizeof(workspace));
    ctx.active_state = &ctx.state;

    mglCoreActivateReplayState(NULL, NULL, NULL);
    mglCoreRestoreLiveActiveState(NULL, NULL);
    CHECK(ctx.active_state == &ctx.state, "NULL calls preserve live state");

    workspace.dirty_bits = 0x12345678u;
    mglCoreActivateReplayState(NULL, &ctx, &workspace);
    CHECK(ctx.active_state == &workspace, "caller workspace becomes active");
    CHECK(mglCtxActiveIsReplayWorkspace(&ctx), "workspace is marked replay");
    CHECK(workspace.dirty_bits == 0x12345678u,
          "caller workspace is not overwritten by a second copy");
    {
        MGLEncodeContext enc;
        memset(&enc, 0, sizeof(enc));
        enc.state = ctx.active_state;
        mglEncodeContextRequireReplayState(&enc, &ctx);
    }

    mglCoreRestoreLiveActiveState(NULL, &ctx);
    CHECK(ctx.active_state == &ctx.state, "teardown restores live state");
    CHECK(!mglCtxActiveIsReplayWorkspace(&ctx), "live state is not replay");
    mglCoreAssertDualProxy(NULL, &ctx);

    ctx.state.dirty_bits = 0xabcdef01u;
    mglCoreActivateReplayState(NULL, &ctx, NULL);
    CHECK(ctx.active_state == &ctx.replay_state,
          "fallback selects the context replay state");
    CHECK(ctx.replay_state.dirty_bits == ctx.state.dirty_bits,
          "fallback snapshots the live state exactly once");
    mglCoreRestoreLiveActiveState(NULL, &ctx);
    CHECK(ctx.active_state == &ctx.state, "fallback teardown restores live state");

    if (failures) {
        return 1;
    }
    printf("test_active_state_restore: ok\n");
    return 0;
}
