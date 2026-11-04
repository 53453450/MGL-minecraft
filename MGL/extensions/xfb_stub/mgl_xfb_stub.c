/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * Fake XFB extension: zero changes to pass/shell; only extension_hooks.h.
 */

#include "mgl_xfb_stub.h"
#include "mgl_extension_hooks.h"

#include <stddef.h>
#include <stdint.h>

enum { kMGLXfbStubExtId = 0x58464201u }; /* 'XFB\1' */

static void xfb_init(GLMContext ctx)
{
    (void)ctx;
}

static void xfb_destroy(GLMContext ctx)
{
    (void)ctx;
}

static uint32_t xfb_enter(GLMContext ctx)
{
    (void)ctx;
    return 0u; /* narrow mask; never DIRTY_ALL */
}

static uint32_t xfb_exit(GLMContext ctx)
{
    (void)ctx;
    return 0u;
}

static int xfb_validate(GLMContext ctx, const struct MGLDrawPlan *in,
                        struct MGLDrawPlan *out)
{
    (void)ctx;
    (void)in;
    (void)out; /* identity: driver keeps input plan */
    return 0;
}

static int xfb_plan(GLMContext ctx, const struct MGLDrawPlan *in,
                    struct MGLEncodePlan *out)
{
    (void)ctx;
    (void)in;
    (void)out;
    return 0; /* stub: no encode plan */
}

static int xfb_encode(struct MGLEncodeContext *enc,
                      const struct MGLEncodePlan *plan)
{
    (void)enc;
    (void)plan;
    return 0;
}

void mglXfbStubRegister(GLMContext ctx)
{
    static const MGLExtensionStateVTable vt = {
        .init = xfb_init,
        .destroy = xfb_destroy,
        .enter_segment = xfb_enter,
        .exit_segment = xfb_exit,
    };
    static const MGLGeometryPipelineStage stage = {
        .name = "xfb_stub",
        .validate = xfb_validate,
        .plan = xfb_plan,
        .encode = xfb_encode,
    };
    mglRegisterExtensionState(ctx, kMGLXfbStubExtId, &vt);
    mglRegisterGeometryStage(ctx, &stage);
}
