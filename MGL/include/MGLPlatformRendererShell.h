/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

#ifndef MGLPlatformRendererShell_h
#define MGLPlatformRendererShell_h

#import <AppKit/AppKit.h>
#import <QuartzCore/QuartzCore.h>
#include <stdint.h>

/* The result record, the operation type and the drawable bridge moved to the
 * C-safe mgl_platform_shell_result.h (the shell class is C++ now, log 210). */
#include "mgl_platform_shell_result.h"

@interface MGLPlatformRendererShell : NSObject
{
    int _swapInterval;
}

@property(nonatomic, strong) NSView *view;
@property(nonatomic, strong) CAMetalLayer *layer;
@property(nonatomic, strong) id<CAMetalDrawable> drawable;

- (instancetype)initWithView:(NSView *)view;
- (int)mglSwapInterval;
- (BOOL)mglShouldSkipPresentForUnlockedSwap;
- (id)mglCreateSystemDefaultDevice;
- (BOOL)mglConfigureMetalLayerWithDevice:(id)device
                    requestedPixelFormat:(uint32_t)requestedPixelFormat
                     actualPixelFormat:(uint32_t *)actualPixelFormat;
- (void)mglDetachMetalLayer;
- (id)mglCaptureDescriptorForDevice:(id)device
                         outputPath:(NSString *)outputPath;
- (BOOL)mglStartCaptureWithDescriptor:(id)descriptor
                                error:(NSError **)error;
- (void)mglStopCapture;
- (id)mglNextDrawable;
- (id)mglDrawableTexture;
- (BOOL)mglHasMetalLayer;
- (CGSize)mglMetalLayerDrawableSize;
- (CGRect)mglMetalLayerFrame;
- (void)mglSetMetalLayerFrame:(CGRect)frame contentsScale:(CGFloat)scale;
- (void)mglSetMetalLayerDrawableSize:(CGSize)size;
- (int)performOperation:(MGLPlatformRendererShellOperation)operation
                context:(void *)context
                 result:(MGLPlatformRendererShellResult *)result;

@end

#endif
