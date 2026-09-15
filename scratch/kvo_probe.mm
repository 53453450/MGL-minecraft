/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * kvo_probe.mm - the window-observation oracle (P0-1 rule 70; log 208).
 *
 * The A/B oracle runs build/test_regression and the CTS battery runs offscreen
 * FBO surfaces; neither ever resizes a real NSWindow, so the platform shell's
 * KVO/notification wiring (main-thread geometry sync, the window observer, the
 * detach path, dealloc) has no coverage from them.  This probe builds the same
 * objects -createMGLRendererAndBindToContext:view: builds, drives them through
 * a resize, and prints what moved:
 *
 *   initial                     applied=200x200   (100x100 view at 2x backing)
 *   NSWindowDidResize posted    applied=400x300   <- the observer ran
 *   backing-properties posted   applied=400x300
 *   same notification for a
 *   foreign object              applied=400x300   <- observers are object-scoped
 *   detach + release            applied=0x0
 *
 * Build (from the repo root):
 *   /usr/bin/clang++ -x objective-c++ -fno-objc-arc -g -O0 \
 *     -isysroot $(xcrun --show-sdk-path) -IMGL/include -IMGL/src \
 *     -IMGL/include/GL -arch arm64 scratch/kvo_probe.mm -Lbuild -lmgl \
 *     -framework Cocoa -framework Metal -framework QuartzCore \
 *     -framework Foundation -o /tmp/kvo_probe
 * Run:  DYLD_LIBRARY_PATH=$PWD/build /tmp/kvo_probe
 *
 * The pristine baseline (8c4e578), the cut before the lifecycle conversion and
 * the cut after it print the same lines once NSLog prefixes and pointers are
 * normalised away.
 */
#import <Cocoa/Cocoa.h>
#import <Metal/Metal.h>
#import "MGLRenderer.h"
#import "MGLRenderer+Lifecycle_Private.h"
#include <stdio.h>
#include "mgl_renderer_ports.h"   /* MGLSizeValue */
/* The apply entry has no header declaration (it is one of the C ports the shell
 * defines); the probe declares it the way the C callers do. */
extern "C" MGLSizeValue mglPlatformShellApplyPendingDrawableSize(void *renderer);

/* The two observer-entry methods are implemented at runtime by the shell. */
@interface MGLRenderer (KVOProbe)
- (void)mglUpdateWindowNotificationObserver;
- (void)mglMainThreadSyncViewGeometry;
@end

/* mglSetMetalLayerFrame:contentsScale: publishes synchronously, so the layer
 * frame is what proves the observer callback ran. */
static void printSize(const char *tag, MGLRenderer *r) {
    /* The applied size is the honest observable: the geometry snapshot only
     * moves when the observer callback ran the sync. */
    MGLSizeValue applied =
        mglPlatformShellApplyPendingDrawableSize((__bridge void *)r);
    CGRect f = [r mglMetalLayerFrame];
    printf("%s applied=%llux%llu frame=%.0fx%.0f hasLayer=%d ready=%d\n", tag,
           (unsigned long long)applied.width, (unsigned long long)applied.height,
           f.size.width, f.size.height,
           [r mglHasMetalLayer] ? 1 : 0, [r mglRendererIsReady] ? 1 : 0);
}

int main(void) {
    @autoreleasepool {
        [NSApplication sharedApplication];
        NSWindow *window = [[NSWindow alloc]
            initWithContentRect:NSMakeRect(0, 0, 100, 100)
                      styleMask:NSWindowStyleMaskBorderless
                        backing:NSBackingStoreBuffered
                          defer:NO];
        NSView *view = [[NSView alloc] initWithFrame:NSMakeRect(0, 0, 100, 100)];
        [window setContentView:view];
        printf("window=%p view=%p\n", (__bridge void *)window, (__bridge void *)view);

        Class rendererClass = NSClassFromString(@"MGLRenderer");
        MGLRenderer *renderer =
            rendererClass ? [[rendererClass alloc] init] : nil;
        renderer.view = view;
        printf("renderer=%p view-bound=%d\n", (__bridge void *)renderer,
               renderer.view == view);

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        uint32_t pf = 0;
        BOOL configured = [renderer mglConfigureMetalLayerWithDevice:device
                                                 requestedPixelFormat:80
                                                  actualPixelFormat:&pf];
        printf("layer configured=%d pf=%u layer-bound=%d\n", configured ? 1 : 0,
               pf, (renderer.layer != nil) ? 1 : 0);

        [renderer mglUpdateWindowNotificationObserver];
        [renderer mglMainThreadSyncViewGeometry];
        printSize("initial", renderer);

        [view setFrameSize:NSMakeSize(200, 150)];
        [[NSNotificationCenter defaultCenter]
            postNotificationName:NSWindowDidResizeNotification object:window];
        printSize("after-resize-notification", renderer);

        [[NSNotificationCenter defaultCenter]
            postNotificationName:NSWindowDidChangeBackingPropertiesNotification
                          object:window];
        printSize("after-backing-notification", renderer);

        /* A notification for another object must not move our geometry. */
        [view setFrameSize:NSMakeSize(320, 240)];
        [[NSNotificationCenter defaultCenter]
            postNotificationName:NSWindowDidResizeNotification
                          object:[[NSObject alloc] init]];
        printSize("after-foreign-notification", renderer);

        /* Detach: the observer must come off without the lazy re-wiring path. */
        [window setContentView:[[NSView alloc] initWithFrame:NSMakeRect(0, 0, 1, 1)]];
        [renderer mglUpdateWindowNotificationObserver];
        printf("detached\n");

        [renderer mglDetachMetalLayer];
        printSize("after-detach-layer", renderer);
        [renderer release];
        printf("released\n");
    }
    printf("PROBE DONE\n");
    return 0;
}
