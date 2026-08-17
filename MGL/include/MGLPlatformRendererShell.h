#ifndef MGLPlatformRendererShell_h
#define MGLPlatformRendererShell_h

#import <AppKit/AppKit.h>
#import <QuartzCore/QuartzCore.h>
#include <stdint.h>

typedef struct MGLPlatformRendererShellResult {
    int32_t status;
    char exception_name[128];
    char exception_reason[512];
} MGLPlatformRendererShellResult;

typedef int (*MGLPlatformRendererShellOperation)(void *context);

@interface MGLPlatformRendererShell : NSObject

@property(nonatomic, strong) NSView *view;
@property(nonatomic, strong) CAMetalLayer *layer;
@property(nonatomic, strong) id<CAMetalDrawable> drawable;

- (instancetype)initWithView:(NSView *)view;
- (int)performOperation:(MGLPlatformRendererShellOperation)operation
                context:(void *)context
                 result:(MGLPlatformRendererShellResult *)result;

@end

#endif
