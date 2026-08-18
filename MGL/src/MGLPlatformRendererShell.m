/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

#import "MGLPlatformRendererShell.h"
#import <Metal/Metal.h>
#include "mgl_render.h"

#include <stdio.h>
#include <string.h>

@implementation MGLPlatformRendererShell

- (instancetype)initWithView:(NSView *)view
{
    self = [super init];
    if (self) {
        _view = view;
    }
    return self;
}

- (id)mglCreateSystemDefaultDevice
{
    return MTLCreateSystemDefaultDevice();
}

- (BOOL)mglConfigureMetalLayerWithDevice:(id)device
                    requestedPixelFormat:(uint32_t)requestedPixelFormat
                     actualPixelFormat:(uint32_t *)actualPixelFormat
{
    if (!device) return NO;

    const uint32_t fallbackPixelFormat = 80u;
    uint32_t pixelFormat = mglRenderMetalLayerPixelFormatIsSupported(
        requestedPixelFormat) ? requestedPixelFormat : fallbackPixelFormat;
    CAMetalLayer *layer = [[CAMetalLayer alloc] init];
    if (!layer) return NO;

    layer.device = device;
    @try {
        layer.pixelFormat = (MTLPixelFormat)pixelFormat;
    } @catch (NSException *exception) {
        NSLog(@"MGL CAMetalLayer invalid pixelFormat=%u requested=%u exception=%@; falling back to BGRA8Unorm",
              pixelFormat, requestedPixelFormat, exception);
        pixelFormat = fallbackPixelFormat;
        layer.pixelFormat = (MTLPixelFormat)pixelFormat;
    }
    layer.opaque = YES;
    layer.framebufferOnly = NO;
    layer.allowsNextDrawableTimeout = YES;
    layer.magnificationFilter = kCAFilterNearest;
    layer.presentsWithTransaction = NO;
    self.layer = layer;

    if (self.view.layer) {
        [self.view.layer addSublayer:layer];
    } else {
        self.view.layer = layer;
    }
    if (actualPixelFormat) *actualPixelFormat = pixelFormat;
    return YES;
}

- (void)mglDetachMetalLayer
{
    self.drawable = nil;
    [self.layer removeFromSuperlayer];
    self.layer = nil;
}

- (id)mglCaptureDescriptorForDevice:(id)device
                         outputPath:(NSString *)outputPath
{
    if (!device || outputPath.length == 0) return nil;
    MTLCaptureDescriptor *descriptor = [[MTLCaptureDescriptor alloc] init];
    descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
    descriptor.outputURL = [NSURL fileURLWithPath:outputPath];
    descriptor.captureObject = device;
    return descriptor;
}

- (BOOL)mglStartCaptureWithDescriptor:(id)descriptor
                                error:(NSError **)error
{
    if (!descriptor) return NO;
    return [MTLCaptureManager.sharedCaptureManager
        startCaptureWithDescriptor:(MTLCaptureDescriptor *)descriptor
        error:error];
}

- (void)mglStopCapture
{
    [MTLCaptureManager.sharedCaptureManager stopCapture];
}

- (id)mglNextDrawable
{
    self.drawable = [self.layer nextDrawable];
    return self.drawable;
}

- (id)mglDrawableTexture
{
    return self.drawable.texture;
}

- (id)mglTextureForDrawable:(id)drawable
{
    return [(id<CAMetalDrawable>)drawable texture];
}

- (BOOL)mglHasMetalLayer
{
    return self.layer != nil;
}

- (CGSize)mglMetalLayerDrawableSize
{
    return self.layer ? self.layer.drawableSize : CGSizeZero;
}

- (CGRect)mglMetalLayerFrame
{
    return self.layer ? self.layer.frame : CGRectZero;
}

- (void)mglSetMetalLayerFrame:(CGRect)frame contentsScale:(CGFloat)scale
{
    if (!self.layer) return;
    self.layer.frame = frame;
    self.layer.contentsScale = scale;
}

- (void)mglSetMetalLayerDrawableSize:(CGSize)size
{
    if (self.layer) self.layer.drawableSize = size;
}

void *mglPlatformRendererShellTextureForDrawable(void *drawable)
{
    if (!drawable) return NULL;
    id<CAMetalDrawable> metalDrawable = (__bridge id<CAMetalDrawable>)drawable;
    return (__bridge void *)metalDrawable.texture;
}

- (int)performOperation:(MGLPlatformRendererShellOperation)operation
                context:(void *)context
                 result:(MGLPlatformRendererShellResult *)result
{
    if (result) memset(result, 0, sizeof(*result));
    if (!operation) return -1;
    @try {
        int status = operation(context);
        if (result) result->status = status;
        return status;
    } @catch (NSException *exception) {
        if (result) {
            result->status = -1;
            snprintf(result->exception_name, sizeof(result->exception_name),
                     "%s", exception.name.UTF8String ?: "NSException");
            snprintf(result->exception_reason, sizeof(result->exception_reason),
                     "%s", exception.reason.UTF8String ?: "unknown");
        }
        return -1;
    }
}

@end
